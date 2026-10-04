"""
Unit and Integration Tests for Facial Recognition Embedding Extraction and Model Isolation.

Covers:
1. InceptionResnetV1 model architecture, 512D output, and L2 unit-norm guarantee.
2. EmbeddingExtractor preprocessing, single/batch extraction, and repeatability.
3. Strict rejection of incompatible inputs (None, empty, non-finite, invalid shapes/channels).
4. Explicit error raising on weight loading failures (strictly NO silent fallbacks to random weights).
5. Cross-model comparison prevention and migration re-enrollment handling in DatabaseManager.
6. Real inference verification when pretrained weights are present in cache.
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from PIL import Image
import torch

from src.models.inception_resnet_v1 import InceptionResnetV1, load_weights, PRETRAINED_URLS, get_checkpoint_dirs
from src.embedding_extraction import EmbeddingExtractor, MultiModelEmbedding
from src.database_manager import DatabaseManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def offline_model():
    """Uninitialized InceptionResnetV1 for fast offline unit tests."""
    model = InceptionResnetV1(pretrained=None, classify=False)
    model.eval()
    return model


@pytest.fixture
def offline_extractor(monkeypatch):
    """EmbeddingExtractor with offline model (pretrained=False) for fast pipeline tests."""
    extractor = EmbeddingExtractor(model_name='vggface2', device='cpu', pretrained=False)
    return extractor


@pytest.fixture
def temp_db_dir():
    """Temporary directory for isolated database migration tests."""
    temp_dir = tempfile.mkdtemp(prefix="test_model_isolation_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. Architecture Tests
# ---------------------------------------------------------------------------

class TestInceptionResnetV1Architecture:
    """Tests verifying InceptionResnetV1 network structure and output invariants."""

    def test_output_shape_and_l2_norm(self, offline_model):
        """Model must output (B, 512) vectors normalized to unit length."""
        batch_size = 3
        dummy_input = torch.randn(batch_size, 3, 160, 160)
        with torch.no_grad():
            output = offline_model(dummy_input)

        assert output.shape == (batch_size, 512)
        norms = torch.norm(output, p=2, dim=1).cpu().numpy()
        assert np.allclose(norms, 1.0, atol=1e-5), f"Expected unit norms, got {norms}"

    def test_repeatability(self, offline_model):
        """Identical inputs must produce bit-for-bit identical embeddings in eval mode."""
        dummy_input = torch.randn(1, 3, 160, 160)
        with torch.no_grad():
            out1 = offline_model(dummy_input)
            out2 = offline_model(dummy_input)

        assert torch.all(out1 == out2)

    def test_classification_mode(self):
        """Classification mode outputs logits corresponding to num_classes."""
        model = InceptionResnetV1(pretrained=None, classify=True, num_classes=100)
        model.eval()
        dummy_input = torch.randn(2, 3, 160, 160)
        with torch.no_grad():
            logits = model(dummy_input)

        assert logits.shape == (2, 100)

    def test_classify_true_without_num_classes_raises(self):
        """classify=True without num_classes or pretrained dataset must raise."""
        with pytest.raises(Exception):
            InceptionResnetV1(pretrained=None, classify=True, num_classes=None)


# ---------------------------------------------------------------------------
# 2. EmbeddingExtractor Preprocessing & Extraction Tests
# ---------------------------------------------------------------------------

class TestEmbeddingExtractorPipeline:
    """Tests for image preprocessing and embedding extraction pipeline."""

    def test_extract_from_valid_rgb_numpy(self, offline_extractor):
        """Valid 160x160 RGB numpy array produces a 512D unit-normalized float32 embedding."""
        img = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        embedding = offline_extractor.extract_embedding(img)

        assert embedding is not None
        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
        assert np.all(np.isfinite(embedding))

    def test_extract_from_grayscale_image(self, offline_extractor):
        """2D Grayscale image is converted to RGB and extracted successfully."""
        gray_img = np.random.randint(0, 255, (160, 160), dtype=np.uint8)
        embedding = offline_extractor.extract_embedding(gray_img)

        assert embedding is not None
        assert embedding.shape == (512,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    def test_extract_from_bgra_image(self, offline_extractor):
        """4-channel BGRA image is converted to RGB and extracted successfully."""
        bgra_img = np.random.randint(0, 255, (160, 160, 4), dtype=np.uint8)
        embedding = offline_extractor.extract_embedding(bgra_img)

        assert embedding is not None
        assert embedding.shape == (512,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    def test_extract_from_pil_image(self, offline_extractor):
        """PIL Image is accepted and processed successfully."""
        pil_img = Image.new('RGB', (160, 160), color=(128, 64, 32))
        embedding = offline_extractor.extract_embedding(pil_img)

        assert embedding is not None
        assert embedding.shape == (512,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)

    def test_batch_extraction(self, offline_extractor):
        """Batch extraction processes multiple images and preserves index ordering."""
        img1 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        results = offline_extractor.extract_batch_embeddings([img1, img2])

        assert len(results) == 2
        assert results[0] is not None and results[0].shape == (512,)
        assert results[1] is not None and results[1].shape == (512,)
        assert np.isclose(np.linalg.norm(results[0]), 1.0, atol=1e-5)
        assert np.isclose(np.linalg.norm(results[1]), 1.0, atol=1e-5)

    def test_mixed_batch_extraction(self, offline_extractor):
        """Batch with valid and invalid inputs returns None for invalid items without failing."""
        valid_img = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        invalid_img = np.array([])  # Empty
        results = offline_extractor.extract_batch_embeddings([valid_img, invalid_img, None])

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is None

    def test_empty_batch_returns_empty_list(self, offline_extractor):
        """Empty input list returns empty list."""
        assert offline_extractor.extract_batch_embeddings([]) == []

    def test_cosine_similarity_calculation(self, offline_extractor):
        """Cosine similarity for identical normalized vectors is 1.0, orthogonal is 0.0."""
        vec1 = np.zeros(512, dtype=np.float32)
        vec1[0] = 1.0
        vec2 = np.zeros(512, dtype=np.float32)
        vec2[0] = 1.0
        vec3 = np.zeros(512, dtype=np.float32)
        vec3[1] = 1.0

        assert np.isclose(offline_extractor.compute_similarity(vec1, vec2), 1.0)
        assert np.isclose(offline_extractor.compute_similarity(vec1, vec3), 0.0)

    def test_distance_metrics(self, offline_extractor):
        """Distance computation correctly supports cosine, euclidean, and manhattan."""
        vec1 = np.zeros(512, dtype=np.float32)
        vec1[0] = 1.0
        vec2 = np.zeros(512, dtype=np.float32)
        vec2[0] = 1.0

        assert np.isclose(offline_extractor.compute_distance(vec1, vec2, metric='cosine'), 0.0)
        assert np.isclose(offline_extractor.compute_distance(vec1, vec2, metric='euclidean'), 0.0)
        assert np.isclose(offline_extractor.compute_distance(vec1, vec2, metric='manhattan'), 0.0)

    def test_model_info_reports_inception_resnet_v1(self, offline_extractor):
        """Model info dict accurately reports architecture, dimensions, and parameter count."""
        info = offline_extractor.get_model_info()
        assert info['architecture'] == 'InceptionResnetV1'
        assert info['embedding_size'] == 512
        assert info['model_parameters'] > 20_000_000


# ---------------------------------------------------------------------------
# 3. Incompatible Input Rejection Tests
# ---------------------------------------------------------------------------

class TestIncompatibleInputRejection:
    """Tests verifying robust rejection of malformed or invalid inputs."""

    def test_none_input_rejected(self, offline_extractor):
        """None image input returns None."""
        assert offline_extractor.extract_embedding(None) is None

    def test_empty_numpy_array_rejected(self, offline_extractor):
        """Empty numpy array returns None."""
        assert offline_extractor.extract_embedding(np.array([])) is None

    def test_non_finite_values_rejected(self, offline_extractor):
        """Image containing NaN or Inf values is safely rejected."""
        nan_img = np.ones((160, 160, 3), dtype=np.float32)
        nan_img[10, 10, 0] = np.nan
        assert offline_extractor.extract_embedding(nan_img) is None

        inf_img = np.ones((160, 160, 3), dtype=np.float32)
        inf_img[20, 20, 1] = np.inf
        assert offline_extractor.extract_embedding(inf_img) is None

    def test_invalid_array_dimensions_rejected(self, offline_extractor):
        """1D and 4D array inputs are rejected."""
        assert offline_extractor.extract_embedding(np.zeros(160)) is None
        assert offline_extractor.extract_embedding(np.zeros((1, 160, 160, 3))) is None

    def test_unsupported_channel_counts_rejected(self, offline_extractor):
        """3D arrays with unsupported channel counts (e.g. 2 or 5 channels) are rejected."""
        assert offline_extractor.extract_embedding(np.zeros((160, 160, 2))) is None
        assert offline_extractor.extract_embedding(np.zeros((160, 160, 5))) is None

    def test_too_small_dimensions_rejected(self, offline_extractor):
        """Micro images (< 10x10) are rejected."""
        assert offline_extractor.extract_embedding(np.zeros((5, 5, 3))) is None
        assert offline_extractor.extract_embedding(Image.new('RGB', (8, 8))) is None


# ---------------------------------------------------------------------------
# 4. Model Loading Failure Handling (Strictly NO Silent Fallbacks)
# ---------------------------------------------------------------------------

class TestModelLoadingFailureHandling:
    """Verifies that model load failures raise explicit exceptions without silent fallbacks."""

    def test_unsupported_model_name_raises_value_error(self):
        """Unsupported model name must raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            EmbeddingExtractor(model_name='unsupported_architecture', pretrained=True)
        assert "Unsupported embedding model" in str(excinfo.value)

    def test_failed_download_raises_runtime_error_no_fallback(self):
        """Network/download failure must raise RuntimeError without substituting random weights."""
        with patch('src.models.inception_resnet_v1.load_weights') as mock_load:
            mock_load.side_effect = RuntimeError("Simulated network download timeout")

            with pytest.raises(RuntimeError) as excinfo:
                EmbeddingExtractor(model_name='vggface2', pretrained=True)

            assert "Failed to load pretrained face recognition model 'vggface2'" in str(excinfo.value)
            assert "random weights will not be substituted" in str(excinfo.value)

    def test_nonexistent_checkpoint_path_raises_runtime_error(self):
        """Explicitly passed invalid checkpoint path must raise RuntimeError."""
        with pytest.raises(RuntimeError) as excinfo:
            EmbeddingExtractor(
                model_name='vggface2',
                pretrained=True,
                checkpoint_path="/nonexistent/path/weights.pt"
            )
        assert "Specified checkpoint_path does not exist" in str(excinfo.value)

    def test_corrupted_checkpoint_raises_runtime_error(self, tmp_path):
        """Corrupted state dict file must raise RuntimeError."""
        corrupt_file = tmp_path / "corrupt_weights.pt"
        corrupt_file.write_bytes(b"CORRUPTED_NOT_A_PYTORCH_CHECKPOINT")

        with pytest.raises(RuntimeError) as excinfo:
            EmbeddingExtractor(
                model_name='vggface2',
                pretrained=True,
                checkpoint_path=str(corrupt_file)
            )
        assert "Model weights are corrupted or incompatible" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 5. Biometric Cross-Model Isolation and Migration Tests
# ---------------------------------------------------------------------------

class TestBiometricCrossModelIsolationAndMigration:
    """Verifies database-level isolation of embedding spaces and safe migration."""

    def test_database_initialization_records_active_model(self, temp_db_dir):
        """Database initialization stores active_embedding_model in system_metadata."""
        db_path = temp_db_dir / "test_meta.db"
        faiss_path = temp_db_dir / "test_meta.faiss"

        db = DatabaseManager(
            db_path=str(db_path),
            faiss_index_path=str(faiss_path),
            embedding_dim=512,
            embedding_model="vggface2"
        )

        stats = db.get_statistics()
        assert stats['active_embedding_model'] == 'vggface2'
        db.close()

    def test_add_embedding_rejects_incompatible_model(self, temp_db_dir):
        """Attempting to add an embedding tagged with a different model space fails."""
        db_path = temp_db_dir / "test_incompat.db"
        faiss_path = temp_db_dir / "test_incompat.faiss"

        db = DatabaseManager(
            db_path=str(db_path),
            faiss_index_path=str(faiss_path),
            embedding_dim=512,
            embedding_model="vggface2"
        )
        db.add_user(user_id="user_1", name="Alice")

        emb = np.random.randn(512).astype(np.float32)
        # Adding with mismatched model_name must be rejected
        emb_id = db.add_embedding(
            user_id="user_1",
            embedding=emb,
            model_name="old_mobilenetv2_random"
        )
        assert emb_id is None
        assert db.get_statistics()['total_embeddings'] == 0
        db.close()

    def test_find_similar_faces_rejects_cross_model_query(self, temp_db_dir):
        """Querying with a model tag incompatible with active index returns empty list."""
        db_path = temp_db_dir / "test_query_incompat.db"
        faiss_path = temp_db_dir / "test_query_incompat.faiss"

        db = DatabaseManager(
            db_path=str(db_path),
            faiss_index_path=str(faiss_path),
            embedding_dim=512,
            embedding_model="vggface2"
        )
        db.add_user(user_id="user_1", name="Alice")
        emb = np.random.randn(512).astype(np.float32)
        db.add_embedding(user_id="user_1", embedding=emb, model_name="vggface2")

        # Search with incompatible model tag must reject cross-model comparison
        hits = db.find_similar_faces(emb, threshold=0.5, model_name="old_random_space")
        assert hits == []
        db.close()

    def test_archive_incompatible_embeddings_and_flag_re_enrollment(self, temp_db_dir):
        """
        Migration test:
        When database contains older embeddings from a prior model:
        1. archive_incompatible_embeddings creates a verified backup.
        2. Incompatible embeddings are marked inactive and removed from FAISS.
        3. Users with no remaining active embeddings are flagged for re-enrollment.
        4. Authentication against archived embeddings is rejected.
        """
        db_path = temp_db_dir / "test_migration.db"
        faiss_path = temp_db_dir / "test_migration.faiss"
        backup_dir = temp_db_dir / "backups"

        # Initialize with legacy model space
        db = DatabaseManager(
            db_path=str(db_path),
            faiss_index_path=str(faiss_path),
            embedding_dim=512,
            embedding_model="mobilenetv2_random",
            backup_dir=str(backup_dir)
        )
        db.add_user(user_id="alice", name="Alice")
        db.add_user(user_id="bob", name="Bob")

        vec_alice = np.random.randn(512).astype(np.float32)
        vec_bob = np.random.randn(512).astype(np.float32)

        # Both enrolled in the old model space
        db.add_embedding(user_id="alice", embedding=vec_alice, model_name="mobilenetv2_random")
        db.add_embedding(user_id="bob", embedding=vec_bob, model_name="mobilenetv2_random")

        assert db.get_statistics()['total_embeddings'] == 2
        assert db.get_statistics()['active_embeddings'] == 2
        assert db.index.ntotal == 2

        # Perform migration to genuine 'vggface2' model
        report = db.archive_incompatible_embeddings(target_model="vggface2")

        assert report['target_model'] == "vggface2"
        assert report['archived_embeddings_count'] == 2
        assert set(report['re_enrollment_users']) == {"alice", "bob"}
        assert Path(report['backup_dir']).exists()

        # Check post-migration database statistics
        stats = db.get_statistics()
        assert stats['active_embedding_model'] == "vggface2"
        assert stats['total_embeddings'] == 2
        assert stats['active_embeddings'] == 0
        assert stats['archived_embeddings'] == 2
        assert stats['faiss_index_size'] == 0

        # Check that users are marked with re_enrollment_required
        alice_data = db.get_user("alice")
        assert alice_data['metadata'].get('re_enrollment_required') is True
        assert "vggface2" in alice_data['metadata'].get('re_enrollment_reason', '')

        # Authenticating with old vector must now fail (no active matches)
        auth_result = db.authenticate_user(vec_alice, threshold=0.1)
        assert auth_result is None

        # Re-enrolling Alice with genuine vggface2 vector succeeds
        vec_alice_vgg = np.random.randn(512).astype(np.float32)
        new_emb_id = db.add_embedding(user_id="alice", embedding=vec_alice_vgg, model_name="vggface2")
        assert new_emb_id is not None
        assert db.get_statistics()['active_embeddings'] == 1
        assert db.index.ntotal == 1

        # Alice now authenticates successfully with new embedding
        match = db.authenticate_user(vec_alice_vgg, threshold=0.8)
        assert match is not None
        assert match['user_id'] == 'alice'

        db.close()


# ---------------------------------------------------------------------------
# 6. Real Inference Verification (Cached Weights)
# ---------------------------------------------------------------------------

class TestRealInferenceWithPretrainedWeights:
    """Verifies genuine pretrained model execution with actual cached weights."""

    def test_real_inference_if_weights_available(self):
        """Runs real inference if the genuine VGGFace2 weights are present in cache."""
        candidate_dirs = get_checkpoint_dirs()
        weights_found = False
        target_path = None

        for d in candidate_dirs:
            path = Path(d) / "20180402-114759-vggface2.pt"
            if path.exists() and path.stat().st_size > 50_000_000:
                weights_found = True
                target_path = str(path)
                break

        if not weights_found:
            pytest.skip("VGGFace2 weights not found in local cache; skipping real weights test.")

        extractor = EmbeddingExtractor(
            model_name='vggface2',
            device='cpu',
            pretrained=True,
            checkpoint_path=target_path
        )

        test_img = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        embedding = extractor.extract_embedding(test_img)

        assert embedding is not None
        assert embedding.shape == (512,)
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)
        assert np.all(np.isfinite(embedding))

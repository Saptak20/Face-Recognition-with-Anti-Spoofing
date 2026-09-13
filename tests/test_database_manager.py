"""
Test Database Manager

Unit and integration tests for the database manager including FAISS operations,
SQLite metadata management, synchronization, consistency validation, and atomic operations.
"""

import pytest
import numpy as np
import threading
import sqlite3
import tempfile
import pickle
import time
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.database_manager import DatabaseManager, FallbackIndex, FAISS_AVAILABLE
if FAISS_AVAILABLE:
    import faiss


class TestFallbackIndex:
    """Test class for FallbackIndex functionality (pure NumPy similarity search)."""

    @pytest.fixture
    def fallback_index(self):
        """Create FallbackIndex instance for testing."""
        return FallbackIndex(dimension=512)

    def test_initialization(self, fallback_index):
        """Test FallbackIndex initialization."""
        assert fallback_index.dimension == 512
        assert len(fallback_index.embeddings) == 0
        assert len(fallback_index.ids) == 0
        assert fallback_index.ntotal == 0

    def test_add_1d_embedding(self, fallback_index):
        """Test adding single 1D embedding."""
        embedding = np.random.randn(512).astype(np.float32)
        fallback_index.add(embedding)

        assert fallback_index.ntotal == 1
        assert len(fallback_index.embeddings) == 1
        assert fallback_index.ids == [0]
        np.testing.assert_allclose(fallback_index.embeddings[0], embedding)

    def test_add_2d_embeddings(self, fallback_index):
        """Test adding batch of 2D embeddings."""
        embeddings = np.random.randn(3, 512).astype(np.float32)
        fallback_index.add(embeddings)

        assert fallback_index.ntotal == 3
        assert len(fallback_index.embeddings) == 3
        assert fallback_index.ids == [0, 1, 2]

    def test_search_similar(self, fallback_index):
        """Test searching similar embeddings in FallbackIndex."""
        # Add 5 random normalized embeddings
        base_embeddings = []
        for i in range(5):
            vec = np.random.randn(512).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            base_embeddings.append(vec)
            fallback_index.add(vec)

        # Query with embedding 0 plus slight noise
        query = base_embeddings[0] + np.random.randn(512).astype(np.float32) * 0.01
        query = query / np.linalg.norm(query)

        distances, indices = fallback_index.search(query, k=3)

        assert distances.shape[1] == 3
        assert indices.shape[1] == 3
        # Embedding 0 should be the closest match
        assert indices[0][0] == 0
        assert distances[0][0] <= distances[0][1] <= distances[0][2]

    def test_search_empty_index(self, fallback_index):
        """Test searching in an empty FallbackIndex."""
        query = np.random.randn(512).astype(np.float32)
        distances, indices = fallback_index.search(query, k=3)

        assert distances.shape == (1, 0) or distances.size == 0
        assert indices.shape == (1, 0) or indices.size == 0

    def test_remove_ids(self, fallback_index):
        """Test removing embeddings by ID in FallbackIndex."""
        for _ in range(3):
            vec = np.random.randn(512).astype(np.float32)
            fallback_index.add(vec)

        assert fallback_index.ids == [0, 1, 2]

        removed = fallback_index.remove_ids(np.array([1]))
        assert removed == 1
        assert fallback_index.ids == [0, 2]
        assert len(fallback_index.embeddings) == 2

        # Removing non-existent ID
        removed_nonexistent = fallback_index.remove_ids(np.array([99]))
        assert removed_nonexistent == 0


class TestDatabaseManagerBasicOperations:
    """Test class for standard DatabaseManager CRUD and authentication operations."""

    @pytest.fixture
    def temp_paths(self, tmp_path):
        """Provide temporary paths for SQLite DB and FAISS index."""
        db_path = str(tmp_path / "test_face_rec.db")
        faiss_path = str(tmp_path / "embeddings" / "test_face_index.faiss")
        return db_path, faiss_path

    @pytest.fixture
    def db_manager(self, temp_paths):
        """Create DatabaseManager instance with temporary storage."""
        db_path, faiss_path = temp_paths
        manager = DatabaseManager(
            db_path=db_path,
            faiss_index_path=faiss_path,
            embedding_dim=512,
            index_type='IndexFlatIP'
        )
        yield manager
        manager.close()

    def test_initialization(self, db_manager, temp_paths):
        """Test DatabaseManager initialization and table creation."""
        db_path, faiss_path = temp_paths
        assert db_manager.db_path == Path(db_path)
        assert db_manager.faiss_index_path == Path(faiss_path)
        assert db_manager.embedding_dim == 512
        assert db_manager.index_type == 'IndexFlatIP'
        assert isinstance(db_manager._lock, type(threading.RLock()))
        assert db_manager.index is not None

        # Verify SQLite tables and indexes exist
        with db_manager._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row['name'] for row in cursor.fetchall()]
            assert 'users' in tables
            assert 'embeddings' in tables
            assert 'auth_logs' in tables

            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row['name'] for row in cursor.fetchall()]
            assert 'idx_users_user_id' in indexes
            assert 'idx_embeddings_user_id' in indexes
            assert 'idx_auth_logs_user_id' in indexes
            assert 'idx_auth_logs_timestamp' in indexes

    def test_add_and_get_user(self, db_manager):
        """Test adding a user and retrieving user info."""
        metadata = {'department': 'Engineering', 'role': 'Lead'}
        success = db_manager.add_user(
            user_id='alice_001',
            name='Alice Smith',
            email='alice@example.com',
            phone='+1-555-0101',
            metadata=metadata
        )
        assert success is True

        user = db_manager.get_user('alice_001')
        assert user is not None
        assert user['user_id'] == 'alice_001'
        assert user['name'] == 'Alice Smith'
        assert user['email'] == 'alice@example.com'
        assert user['phone'] == '+1-555-0101'
        assert user['is_active'] is True
        assert user['metadata'] == metadata

    def test_add_user_duplicate_fails(self, db_manager):
        """Test that duplicate user_id is rejected."""
        assert db_manager.add_user('user_dup', 'User One') is True
        assert db_manager.add_user('user_dup', 'User Two') is False

    def test_get_user_nonexistent(self, db_manager):
        """Test getting non-existent user returns None."""
        assert db_manager.get_user('ghost_user') is None

    def test_add_embedding_dimension_mismatch(self, db_manager):
        """Test adding embedding with wrong dimension is rejected."""
        db_manager.add_user('user_dim', 'Dim Test')
        wrong_emb = np.random.randn(256).astype(np.float32)
        emb_id = db_manager.add_embedding('user_dim', wrong_emb)
        assert emb_id is None

    def test_add_embedding_success(self, db_manager):
        """Test successfully adding an embedding."""
        db_manager.add_user('user_emb', 'Emb User')
        vec = np.random.randn(512).astype(np.float32)
        emb_id = db_manager.add_embedding('user_emb', vec, quality_score=0.94)

        assert emb_id is not None
        assert isinstance(emb_id, str)
        assert db_manager.index.ntotal == 1

        # Check SQLite record
        with db_manager._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM embeddings WHERE embedding_id = ?", (emb_id,))
            row = cursor.fetchone()
            assert row is not None
            assert row['user_id'] == 'user_emb'
            assert row['faiss_id'] == 0
            assert abs(row['quality_score'] - 0.94) < 1e-4

    def test_find_similar_faces_and_authenticate(self, db_manager):
        """Test finding similar faces and authenticating an active user."""
        db_manager.add_user('user_auth', 'Auth User', email='auth@example.com')
        db_manager.add_user('user_other', 'Other User', email='other@example.com')

        vec1 = np.random.randn(512).astype(np.float32)
        vec1_norm = vec1 / np.linalg.norm(vec1)
        db_manager.add_embedding('user_auth', vec1_norm, quality_score=0.96)

        vec2 = np.random.randn(512).astype(np.float32)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        db_manager.add_embedding('user_other', vec2_norm, quality_score=0.91)

        # Search with similar vector to user_auth
        query = vec1_norm + np.random.randn(512).astype(np.float32) * 0.01
        query_norm = query / np.linalg.norm(query)

        similar_faces = db_manager.find_similar_faces(query_norm, k=2, threshold=0.7)
        assert len(similar_faces) >= 1
        assert similar_faces[0]['user_id'] == 'user_auth'
        assert similar_faces[0]['similarity'] > 0.8

        # Test authenticate_user method
        auth_result = db_manager.authenticate_user(query_norm, threshold=0.7)
        assert auth_result is not None
        assert auth_result['user_id'] == 'user_auth'
        assert auth_result['name'] == 'Auth User'
        assert auth_result['email'] == 'auth@example.com'

    def test_authenticate_inactive_user_returns_none(self, db_manager):
        """Test that inactive users are not authenticated."""
        db_manager.add_user('user_inactive', 'Inactive User')
        vec = np.random.randn(512).astype(np.float32)
        vec_norm = vec / np.linalg.norm(vec)
        db_manager.add_embedding('user_inactive', vec_norm)

        # Mark user as inactive in database
        with db_manager._get_db_connection() as conn:
            conn.execute("UPDATE users SET is_active = 0 WHERE user_id = 'user_inactive'")
            conn.commit()

        user = db_manager.get_user('user_inactive')
        assert user['is_active'] is False

        auth_result = db_manager.authenticate_user(vec_norm, threshold=0.7)
        assert auth_result is None

    def test_authenticate_user_no_match(self, db_manager):
        """Test authentication returns None when no embeddings meet the threshold."""
        db_manager.add_user('user_known', 'Known User')
        vec = np.random.randn(512).astype(np.float32)
        db_manager.add_embedding('user_known', vec)

        # Use an orthogonal query with high threshold
        orthogonal_vec = np.random.randn(512).astype(np.float32)
        auth_result = db_manager.authenticate_user(orthogonal_vec, threshold=0.999)
        assert auth_result is None

    def test_log_authentication_and_get_statistics(self, db_manager):
        """Test logging authentication attempts and computing statistics."""
        db_manager.add_user('user_stat_1', 'User Stat 1')
        db_manager.add_user('user_stat_2', 'User Stat 2')
        db_manager.add_embedding('user_stat_1', np.random.randn(512).astype(np.float32), quality_score=0.9)
        db_manager.add_embedding('user_stat_2', np.random.randn(512).astype(np.float32), quality_score=0.8)

        # Log attempts
        log_success = db_manager.log_authentication(
            user_id='user_stat_1',
            success=True,
            confidence_score=0.88,
            liveness_score=0.91,
            deepfake_score=0.90,
            ip_address='127.0.0.1',
            metadata={'client': 'pytest'}
        )
        assert log_success is True

        db_manager.log_authentication(
            user_id=None,
            success=False,
            confidence_score=0.2,
            liveness_score=0.3,
            deepfake_score=0.4,
            ip_address='192.168.1.1'
        )

        stats = db_manager.get_statistics()
        assert stats['total_users'] == 2
        assert stats['active_users'] == 2
        assert stats['total_embeddings'] == 2
        assert stats['total_authentications'] == 2
        assert stats['successful_authentications'] == 1
        assert stats['success_rate_percent'] == 50.0
        assert stats['faiss_index_size'] == 2


class TestFAISSSynchronizationAndRebuild:
    """
    Phase 1 Task 2 Verification:
    Tests FAISS ↔ SQLite synchronization, user deletion, index rebuilding,
    faiss_id remapping, zero-vector handling, atomic saving, and backup validation.
    """

    @pytest.fixture
    def temp_env(self, tmp_path):
        """Setup isolated paths for database and index."""
        db_path = str(tmp_path / "sync_test.db")
        faiss_path = str(tmp_path / "embeddings" / "sync_index.faiss")
        backup_dir = str(tmp_path / "backups")
        return db_path, faiss_path, backup_dir

    @pytest.fixture
    def db(self, temp_env):
        """Instantiate DatabaseManager."""
        db_path, faiss_path, _ = temp_env
        manager = DatabaseManager(
            db_path=db_path,
            faiss_index_path=faiss_path,
            embedding_dim=512,
            index_type='IndexFlatIP'
        )
        yield manager
        manager.close()

    def test_requirement_1_add_embeddings_increases_faiss_count(self, db):
        """1. Add embedding -> FAISS vector count increases."""
        db.add_user('user_a', 'Alice')
        db.add_user('user_b', 'Bob')

        assert db.index.ntotal == 0

        db.add_embedding('user_a', np.random.randn(512).astype(np.float32))
        assert db.index.ntotal == 1

        db.add_embedding('user_a', np.random.randn(512).astype(np.float32))
        assert db.index.ntotal == 2

        db.add_embedding('user_b', np.random.randn(512).astype(np.float32))
        assert db.index.ntotal == 3

        # Verify SQLite faiss_id sequential mapping
        with db._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT faiss_id FROM embeddings ORDER BY id")
            faiss_ids = [r['faiss_id'] for r in cursor.fetchall()]
            assert faiss_ids == [0, 1, 2]

    def test_requirements_2_3_4_delete_user_rebuilds_and_remaps_consistently(self, db):
        """
        2. Delete user -> their embeddings disappear from FAISS.
        3. Remaining embeddings are correctly remapped to new faiss_id values.
        4. Remaining users still resolve to the correct vectors after rebuild.
        """
        # Create 3 users
        db.add_user('alice', 'Alice')
        db.add_user('bob', 'Bob')
        db.add_user('charlie', 'Charlie')

        # Create distinct normalized test vectors
        vec_alice_1 = np.random.randn(512).astype(np.float32)
        vec_alice_1 /= np.linalg.norm(vec_alice_1)

        vec_alice_2 = np.random.randn(512).astype(np.float32)
        vec_alice_2 /= np.linalg.norm(vec_alice_2)

        vec_bob = np.random.randn(512).astype(np.float32)
        vec_bob /= np.linalg.norm(vec_bob)

        vec_charlie = np.random.randn(512).astype(np.float32)
        vec_charlie /= np.linalg.norm(vec_charlie)

        # Add embeddings in order: alice(0), bob(1), alice(2), charlie(3)
        db.add_embedding('alice', vec_alice_1)
        db.add_embedding('bob', vec_bob)
        db.add_embedding('alice', vec_alice_2)
        db.add_embedding('charlie', vec_charlie)

        assert db.index.ntotal == 4

        # Delete user 'bob' (who was at faiss_id 1)
        delete_success = db.delete_user('bob')
        assert delete_success is True

        # Requirement 2: bob's embeddings disappear from FAISS
        assert db.index.ntotal == 3

        # Check SQLite
        with db._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM users WHERE user_id = 'bob'")
            assert cursor.fetchone()['cnt'] == 0

            cursor.execute("SELECT COUNT(*) as cnt FROM embeddings WHERE user_id = 'bob'")
            assert cursor.fetchone()['cnt'] == 0

            # Requirement 3: remaining embeddings remapped to sequential faiss_ids [0, 1, 2]
            cursor.execute("SELECT user_id, faiss_id FROM embeddings ORDER BY faiss_id")
            rows = cursor.fetchall()
            remapped_faiss_ids = [r['faiss_id'] for r in rows]
            assert remapped_faiss_ids == [0, 1, 2]

        # Requirement 4: Remaining users still resolve to the correct vectors after rebuild
        # Alice 1
        res_alice_1 = db.find_similar_faces(vec_alice_1, k=1, threshold=0.95)
        assert len(res_alice_1) == 1
        assert res_alice_1[0]['user_id'] == 'alice'

        # Alice 2
        res_alice_2 = db.find_similar_faces(vec_alice_2, k=1, threshold=0.95)
        assert len(res_alice_2) == 1
        assert res_alice_2[0]['user_id'] == 'alice'

        # Charlie
        res_charlie = db.find_similar_faces(vec_charlie, k=1, threshold=0.95)
        assert len(res_charlie) == 1
        assert res_charlie[0]['user_id'] == 'charlie'

        # Bob should no longer match
        res_bob = db.find_similar_faces(vec_bob, k=1, threshold=0.95)
        if len(res_bob) > 0:
            assert res_bob[0]['user_id'] != 'bob'

    def test_requirement_5_delete_final_user_results_in_valid_zero_vector_index(self, db):
        """5. Delete final user -> valid zero-vector FAISS index."""
        db.add_user('user_solo', 'Solo User')
        vec = np.random.randn(512).astype(np.float32)
        db.add_embedding('user_solo', vec)

        assert db.index.ntotal == 1

        # Delete the only user
        deleted = db.delete_user('user_solo')
        assert deleted is True

        # Check index state
        assert db.index.ntotal == 0
        assert db._validate_faiss_index() is True

        # Searching zero-vector index does not crash and returns empty list
        query = np.random.randn(512).astype(np.float32)
        results = db.find_similar_faces(query, k=5)
        assert results == []

        # Adding a new user and embedding afterwards succeeds cleanly
        db.add_user('user_next', 'Next User')
        new_emb_id = db.add_embedding('user_next', query)
        assert new_emb_id is not None
        assert db.index.ntotal == 1

        with db._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT faiss_id FROM embeddings WHERE user_id = 'user_next'")
            assert cursor.fetchone()['faiss_id'] == 0

    def test_requirements_6_7_backup_database_integrity_and_dimensions(self, db, temp_env):
        """
        6. FAISS backup can be loaded successfully.
        7. Backup dimension and vector count is correct.
        """
        _, _, backup_dir = temp_env

        # Populate with test users and embeddings
        for i in range(3):
            uid = f'backup_user_{i}'
            db.add_user(uid, f'Backup User {i}')
            vec = np.random.randn(512).astype(np.float32)
            db.add_embedding(uid, vec)

        assert db.index.ntotal == 3

        # Create backup
        backup_success = db.backup_database(backup_dir)
        assert backup_success is True

        # Find the created backup files
        backup_path = Path(backup_dir)
        sqlite_backups = list(backup_path.glob("face_recognition_*.db"))
        faiss_backups = list(backup_path.glob("face_index_*.faiss"))

        assert len(sqlite_backups) == 1
        assert len(faiss_backups) == 1

        faiss_backup_file = faiss_backups[0]
        sqlite_backup_file = sqlite_backups[0]

        # Requirement 6 & 7: Verify FAISS backup load and properties
        if FAISS_AVAILABLE:
            loaded_index = faiss.read_index(str(faiss_backup_file))
            assert loaded_index.d == 512
            assert loaded_index.ntotal == 3
        else:
            with open(faiss_backup_file, 'rb') as f:
                loaded_index = pickle.load(f)
            assert loaded_index.dimension == 512
            assert loaded_index.ntotal == 3

        # Verify SQLite backup contains corresponding data
        backup_conn = sqlite3.connect(str(sqlite_backup_file))
        cursor = backup_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        assert cursor.fetchone()[0] == 3
        backup_conn.close()

    def test_requirement_8_validate_faiss_index_detects_mismatches(self, db):
        """8. FAISS/index consistency validation detects mismatches where practical."""
        db.add_user('user_valid', 'Valid User')
        vec = np.random.randn(512).astype(np.float32)
        db.add_embedding('user_valid', vec)

        # Baseline: index is valid
        assert db._validate_faiss_index() is True

        # Tamper 1: Count mismatch (insert orphan row in SQLite)
        with db._get_db_connection() as conn:
            conn.execute(
                "INSERT INTO embeddings (user_id, embedding_id, faiss_id, embedding_vector, quality_score) "
                "VALUES ('user_valid', 'fake-emb-id', 1, ?, 0.9)",
                (pickle.dumps(vec),)
            )
            conn.commit()

        # FAISS ntotal is 1, but SQLite count is 2 -> must return False
        assert db._validate_faiss_index() is False

        # Remove the orphan row
        with db._get_db_connection() as conn:
            conn.execute("DELETE FROM embeddings WHERE embedding_id = 'fake-emb-id'")
            conn.commit()

        assert db._validate_faiss_index() is True

        # Tamper 2: Invalid faiss_id range in SQLite
        with db._get_db_connection() as conn:
            conn.execute("UPDATE embeddings SET faiss_id = 999 WHERE user_id = 'user_valid'")
            conn.commit()

        # faiss_id 999 >= count 1 -> must return False
        assert db._validate_faiss_index() is False

    def test_reentrant_lock_prevents_deadlock_on_rebuild(self, db):
        """Verify RLock allows nested locking without deadlocking during delete_user/rebuild."""
        db.add_user('deadlock_user', 'Deadlock Test')
        db.add_embedding('deadlock_user', np.random.randn(512).astype(np.float32))

        # Acquire lock externally, then call delete_user (which also acquires lock)
        acquired = db._lock.acquire(timeout=2.0)
        assert acquired is True
        try:
            # If self._lock was a regular Lock(), this call would deadlock here
            result = db.delete_user('deadlock_user')
            assert result is True
        finally:
            db._lock.release()

        assert db.index.ntotal == 0

    def test_atomic_persistence_and_no_tmp_leftover(self, db, temp_env):
        """Verify _save_faiss_index replaces atomically and cleans up temporary file."""
        _, faiss_path, _ = temp_env
        db.add_user('atomic_user', 'Atomic Test')
        db.add_embedding('atomic_user', np.random.randn(512).astype(np.float32))

        faiss_file = Path(faiss_path)
        temp_file = faiss_file.with_suffix('.faiss.tmp')

        assert faiss_file.exists()
        assert not temp_file.exists()


class TestDatabaseIntegrationWorkflow:
    """Integration workflow test simulating multiple concurrent operations."""

    @pytest.fixture
    def int_db(self, tmp_path):
        """Create integration test database."""
        db_path = str(tmp_path / "integration.db")
        faiss_path = str(tmp_path / "integration.faiss")
        manager = DatabaseManager(
            db_path=db_path,
            faiss_index_path=faiss_path,
            embedding_dim=512
        )
        yield manager
        manager.close()

    def test_full_lifecycle_workflow(self, int_db):
        """Test full user lifecycle: registration, search, authentication, deletion, rebuild."""
        user_ids = ['alice', 'bob', 'charlie', 'david']
        user_vectors = {}

        # 1. Register users and embeddings
        for uid in user_ids:
            assert int_db.add_user(uid, f'{uid.capitalize()} Name', email=f'{uid}@test.com') is True
            vec = np.random.randn(512).astype(np.float32)
            vec /= np.linalg.norm(vec)
            user_vectors[uid] = vec
            assert int_db.add_embedding(uid, vec, quality_score=0.95) is not None

        assert int_db.index.ntotal == 4

        # 2. Authenticate each user
        for uid in user_ids:
            auth = int_db.authenticate_user(user_vectors[uid], threshold=0.9)
            assert auth is not None
            assert auth['user_id'] == uid

        # 3. Delete middle user 'bob'
        assert int_db.delete_user('bob') is True
        assert int_db.index.ntotal == 3

        # 4. Verify remaining users still authenticate
        for uid in ['alice', 'charlie', 'david']:
            auth = int_db.authenticate_user(user_vectors[uid], threshold=0.9)
            assert auth is not None
            assert auth['user_id'] == uid

        # Bob fails
        assert int_db.authenticate_user(user_vectors['bob'], threshold=0.9) is None

        # 5. Delete remaining users
        for uid in ['alice', 'charlie', 'david']:
            assert int_db.delete_user(uid) is True

        assert int_db.index.ntotal == 0
        assert int_db._validate_faiss_index() is True
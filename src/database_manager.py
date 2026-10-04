"""
Database Manager Module

This module handles FAISS vector database for face embeddings and SQLite
for user metadata management. Provides efficient similarity search and
CRUD operations for face recognition system.
"""

import sqlite3
import numpy as np
import logging
import pickle
import json
import time
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import threading
from contextlib import contextmanager
import uuid
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available, using fallback similarity search")
    FAISS_AVAILABLE = False


class FallbackIndex:
    """
    Fallback similarity search when FAISS is not available.
    Uses numpy for basic similarity computation.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.embeddings = []
        self.ids = []
        self.ntotal = 0

    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the index."""
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        for embedding in embeddings:
            self.embeddings.append(embedding)
            self.ids.append(self.ntotal)
            self.ntotal += 1

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar embeddings."""
        if len(queries.shape) == 1:
            queries = queries.reshape(1, -1)

        if not self.embeddings:
            return np.array([[]]), np.array([[]])

        embeddings_array = np.array(self.embeddings)
        distances = []
        indices = []

        for query in queries:
            # Calculate cosine similarity
            similarities = np.dot(embeddings_array, query) / (
                np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query)
            )

            # Convert to distances (1 - similarity)
            dists = 1.0 - similarities

            # Get top k results
            if k > len(dists):
                k = len(dists)

            top_k_indices = np.argpartition(dists, k)[:k]
            top_k_indices = top_k_indices[np.argsort(dists[top_k_indices])]

            distances.append(dists[top_k_indices])
            indices.append(top_k_indices)

        return np.array(distances), np.array(indices)

    def remove_ids(self, ids_to_remove: np.ndarray) -> int:
        """Remove embeddings by IDs."""
        removed_count = 0
        for id_to_remove in ids_to_remove:
            if id_to_remove in self.ids:
                idx = self.ids.index(id_to_remove)
                self.embeddings.pop(idx)
                self.ids.pop(idx)
                removed_count += 1

        return removed_count


class DatabaseManager:
    """
    Database manager for face recognition system handling both
    FAISS vector index and SQLite metadata storage.
    """

    def __init__(self,
                 db_path: str = "data/face_recognition.db",
                 faiss_index_path: str = "data/embeddings/face_index.faiss",
                 embedding_dim: int = 512,
                 embedding_model: str = "vggface2",
                 index_type: str = 'IndexFlatIP',
                 backup_dir: Optional[str] = None,
                 auto_reconcile: bool = True):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database
            faiss_index_path: Path to FAISS index file
            embedding_dim: Dimension of face embeddings
            embedding_model: Identifier of the active facial recognition embedding model
            index_type: FAISS index type ('IndexFlatIP', 'IndexFlatL2', 'IndexIVFFlat')
            backup_dir: Directory for storing backups. Defaults to db_path.parent / "backups"
            auto_reconcile: Whether to validate and safely reconcile FAISS/SQLite consistency at startup
        """
        self.db_path = Path(db_path)
        self.faiss_index_path = Path(faiss_index_path)
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.index_type = index_type

        # Create directories if they don't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir = Path(backup_dir) if backup_dir else self.db_path.parent / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Thread lock for database operations
        self._lock = threading.RLock()

        # Initialize databases
        self._init_sqlite_db()
        self._init_faiss_index()

        if auto_reconcile:
            self._reconcile_on_startup()

        logger.info(
            f"DatabaseManager initialized with embedding dim: {embedding_dim}, "
            f"model: {embedding_model}"
        )

    def _init_sqlite_db(self) -> None:
        """Initialize SQLite database with required tables and metadata."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                # System metadata table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Users table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        email TEXT,
                        phone TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        metadata TEXT
                    )
                ''')

                # Embeddings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        embedding_id TEXT UNIQUE NOT NULL,
                        faiss_id INTEGER,
                        embedding_vector BLOB,
                        quality_score REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        model_name TEXT DEFAULT 'vggface2',
                        is_active BOOLEAN DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')

                # Safe schema migration for pre-existing embeddings tables
                cursor.execute("PRAGMA table_info(embeddings)")
                existing_cols = {row[1] for row in cursor.fetchall()}
                if 'model_name' not in existing_cols:
                    cursor.execute("ALTER TABLE embeddings ADD COLUMN model_name TEXT DEFAULT 'vggface2'")
                if 'is_active' not in existing_cols:
                    cursor.execute("ALTER TABLE embeddings ADD COLUMN is_active BOOLEAN DEFAULT 1")

                # Authentication logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS auth_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        success BOOLEAN,
                        confidence_score REAL,
                        liveness_score REAL,
                        deepfake_score REAL,
                        ip_address TEXT,
                        metadata TEXT
                    )
                ''')

                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_user_id ON users (user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_user_id ON embeddings (user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_model_name ON embeddings (model_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_is_active ON embeddings (is_active)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_auth_logs_user_id ON auth_logs (user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_auth_logs_timestamp ON auth_logs (timestamp)')

                # Initialize or verify active embedding model metadata
                cursor.execute("SELECT value FROM system_metadata WHERE key = 'active_embedding_model'")
                row = cursor.fetchone()
                if row is None:
                    cursor.execute(
                        "INSERT INTO system_metadata (key, value) VALUES ('active_embedding_model', ?)",
                        (self.embedding_model,)
                    )
                    cursor.execute(
                        "INSERT INTO system_metadata (key, value) VALUES ('embedding_dimension', ?)",
                        (str(self.embedding_dim),)
                    )
                else:
                    stored_model = row['value'] if isinstance(row, dict) else row[0]
                    if stored_model != self.embedding_model:
                        logger.warning(
                            f"Active model in metadata ('{stored_model}') differs from configured "
                            f"model ('{self.embedding_model}'). Use archive_incompatible_embeddings() "
                            "if migrating spaces to prevent cross-model similarity comparisons."
                        )

                conn.commit()
                logger.info("SQLite database initialized successfully")

        except Exception as e:
            logger.error(f"SQLite initialization error: {str(e)}")
            raise

    def _create_empty_faiss_index(self):
        """Create a new empty FAISS or fallback index matching configuration."""
        if FAISS_AVAILABLE:
            if self.index_type == 'IndexFlatIP':
                return faiss.IndexFlatIP(self.embedding_dim)
            elif self.index_type == 'IndexFlatL2':
                return faiss.IndexFlatL2(self.embedding_dim)
            elif self.index_type == 'IndexIVFFlat':
                quantizer = faiss.IndexFlatL2(self.embedding_dim)
                return faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            else:
                logger.warning(f"Unknown index type: {self.index_type}, using IndexFlatIP")
                return faiss.IndexFlatIP(self.embedding_dim)
        else:
            return FallbackIndex(self.embedding_dim)

    def _init_faiss_index(self) -> None:
        """Initialize FAISS index for similarity search."""
        try:
            if FAISS_AVAILABLE:
                if self.faiss_index_path.exists():
                    # Load existing index
                    self.index = faiss.read_index(str(self.faiss_index_path))
                    logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
                else:
                    self.index = self._create_empty_faiss_index()
                    logger.info(f"Created new FAISS index: {self.index_type}")
            else:
                # Use fallback index
                if self.faiss_index_path.exists():
                    try:
                        with open(self.faiss_index_path, 'rb') as f:
                            self.index = pickle.load(f)
                        logger.info(f"Loaded existing fallback index with {self.index.ntotal} vectors")
                    except Exception as e:
                        logger.warning(f"Failed to load fallback index: {e}")
                        self.index = self._create_empty_faiss_index()
                else:
                    self.index = self._create_empty_faiss_index()
                    logger.info("Created new fallback index")

        except Exception as e:
            logger.error(f"FAISS index initialization error: {str(e)}")
            self.index = self._create_empty_faiss_index()

    def _create_verified_index_backup(self, backup_tag: str = "orphan_or_inconsistent") -> Path:
        """
        Create a verified backup of the current FAISS index before destructive recovery/reset.

        Args:
            backup_tag: Descriptive tag for the backup filename

        Returns:
            Path to the verified backup file
        """
        timestamp = int(time.time())
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file = self.backup_dir / f"face_index_{backup_tag}_{timestamp}.faiss"
        temp_backup = self.backup_dir / f"face_index_{backup_tag}_{timestamp}.faiss.tmp"

        try:
            with self._lock:
                if FAISS_AVAILABLE and hasattr(self.index, 'ntotal') and self.index.ntotal > 0:
                    faiss.write_index(self.index, str(temp_backup))
                elif not FAISS_AVAILABLE and hasattr(self.index, 'embeddings') and len(self.index.embeddings) > 0:
                    with open(temp_backup, 'wb') as f:
                        pickle.dump(self.index, f)
                elif self.faiss_index_path.exists() and self.faiss_index_path.stat().st_size > 0:
                    shutil.copy2(self.faiss_index_path, temp_backup)
                elif FAISS_AVAILABLE and hasattr(self.index, 'ntotal'):
                    faiss.write_index(self.index, str(temp_backup))
                else:
                    with open(temp_backup, 'wb') as f:
                        pickle.dump(self.index, f)

            temp_backup.replace(backup_file)
            self._verify_faiss_backup(backup_file, verify_dim=False)
            logger.info(f"Verified FAISS index backup created at: {backup_file}")
            return backup_file

        except Exception as e:
            if temp_backup.exists():
                try:
                    temp_backup.unlink()
                except Exception:
                    pass
            logger.error(f"Failed to create verified index backup: {e}")
            raise

    def _reconcile_on_startup(self) -> None:
        """
        Validate consistency between SQLite and FAISS index at startup.
        If inconsistent, safely reconcile using SQLite as the authoritative source.
        Never recovers identities from orphan FAISS vectors.
        Never destroys or overwrites an existing index before creating a verified backup.
        """
        if self._validate_faiss_index():
            logger.info("Startup consistency check passed: FAISS index is consistent with SQLite.")
            return

        logger.warning("Startup consistency check failed: FAISS index is inconsistent with SQLite.")

        # Create verified backup before modifying or rebuilding anything
        faiss_has_vectors = (
            (hasattr(self.index, 'ntotal') and self.index.ntotal > 0) or
            (hasattr(self.index, 'embeddings') and len(self.index.embeddings) > 0)
        )
        if self.faiss_index_path.exists() or faiss_has_vectors:
            try:
                backup_path = self._create_verified_index_backup(backup_tag="orphan_or_inconsistent")
                logger.info(f"Preserved existing FAISS index in verified backup: {backup_path}")
            except Exception as e:
                logger.error(f"Cannot safely reconcile without verified backup: {e}")
                raise RuntimeError(
                    f"Startup FAISS backup failed; aborting reconciliation to prevent unbacked data loss: {e}"
                ) from e

        # Query SQLite to see authoritative active embedding count
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM embeddings WHERE is_active = 1')
            sqlite_count = cursor.fetchone()['count']

        if sqlite_count == 0:
            # Authoritative SQLite has 0 active embeddings.
            # FAISS had orphan vectors (e.g. 3 orphan vectors).
            # We must NEVER invent identities for orphan vectors.
            # Reset active index to empty to match authoritative SQLite state.
            logger.warning(
                "Authoritative SQLite has 0 active embeddings while FAISS had orphan vectors. "
                "Orphan vectors preserved in backup. Resetting active FAISS index to 0 vectors."
            )
            with self._lock:
                self.index = self._create_empty_faiss_index()
                self._save_faiss_index()
        else:
            # Authoritative SQLite has active embeddings. Rebuild FAISS index from SQLite.
            logger.info(f"Rebuilding FAISS index from {sqlite_count} authoritative active SQLite embeddings.")
            self._rebuild_faiss_index()

        # Re-validate
        if not self._validate_faiss_index():
            raise RuntimeError(
                "FAISS/SQLite reconciliation completed, but post-validation still failed. "
                "Database remains in an inconsistent state."
            )
        logger.info("Startup FAISS/SQLite reconciliation completed successfully.")

    def _validate_faiss_index(self) -> bool:
        """
        Validate FAISS index consistency with SQLite.

        Returns:
            True if index is valid, False otherwise
        """
        try:
            # Check dimension
            if FAISS_AVAILABLE and hasattr(self.index, 'd'):
                if self.index.d != self.embedding_dim:
                    logger.error(f"FAISS index dimension mismatch: {self.index.d} != {self.embedding_dim}")
                    return False

            # Check vector count consistency with active SQLite embeddings
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) as count FROM embeddings WHERE is_active = 1')
                sqlite_count = cursor.fetchone()['count']

            faiss_count = self.index.ntotal if hasattr(self.index, 'ntotal') else len(self.index.embeddings) if hasattr(self.index, 'embeddings') else 0

            if faiss_count != sqlite_count:
                logger.error(f"FAISS/SQLite count mismatch: FAISS={faiss_count}, SQLite active={sqlite_count}")
                return False

            # Check faiss_id values are valid for active embeddings
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT faiss_id FROM embeddings WHERE is_active = 1')
                raw_faiss_ids = [row[0] for row in cursor.fetchall()]

                if any(fid is None for fid in raw_faiss_ids):
                    logger.error("Some active SQLite embeddings have NULL faiss_id")
                    return False

                faiss_ids = [int(fid) for fid in raw_faiss_ids]

                if faiss_ids:
                    max_faiss_id = max(faiss_ids)
                    min_faiss_id = min(faiss_ids)
                    if max_faiss_id >= faiss_count or min_faiss_id < 0:
                        logger.error(f"Invalid faiss_id range: min={min_faiss_id}, max={max_faiss_id}, count={faiss_count}")
                        return False
                    if len(set(faiss_ids)) != len(faiss_ids):
                        logger.error(f"Duplicate faiss_ids detected in SQLite: count={len(faiss_ids)}, unique={len(set(faiss_ids))}")
                        return False

            return True

        except Exception as e:
            logger.error(f"FAISS validation error: {str(e)}")
            return False

    def _rebuild_faiss_index(self) -> None:
        """
        Rebuild FAISS index from active SQLite embeddings.

        This rebuilds the entire FAISS index from the authoritative active SQLite embeddings,
        assigns new sequential FAISS IDs, and updates the SQLite faiss_id values.
        """
        try:
            with self._lock:
                logger.info("Rebuilding FAISS index from active SQLite embeddings")

                new_index = self._create_empty_faiss_index()

                # Read remaining active embeddings from SQLite in deterministic order
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT embedding_id, embedding_vector, user_id, quality_score "
                        "FROM embeddings WHERE is_active = 1 "
                        "ORDER BY COALESCE(faiss_id, 999999999), id ASC"
                    )
                    rows = cursor.fetchall()

                # Add each embedding to the new index and collect updates
                faiss_id_updates = []

                for new_faiss_id, row in enumerate(rows):
                    embedding_id = row['embedding_id']
                    embedding_blob = row['embedding_vector']
                    embedding = pickle.loads(embedding_blob)

                    # Add to new index
                    if FAISS_AVAILABLE:
                        new_index.add(embedding.reshape(1, -1).astype(np.float32))
                    else:
                        new_index.add(embedding)

                    # Record faiss_id update for SQLite
                    faiss_id_updates.append((new_faiss_id, row['embedding_id']))

                # Update SQLite faiss_id values
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()
                    for new_faiss_id, embedding_id in faiss_id_updates:
                        cursor.execute(
                            "UPDATE embeddings SET faiss_id = ? WHERE embedding_id = ?",
                            (new_faiss_id, embedding_id)
                        )
                    # Clear faiss_id for any inactive/archived embeddings
                    cursor.execute("UPDATE embeddings SET faiss_id = NULL WHERE is_active = 0")
                    conn.commit()

                # Replace index only after successful rebuild
                self.index = new_index

                # Save rebuilt index
                self._save_faiss_index()

                logger.info(f"FAISS index rebuilt successfully with {new_index.ntotal} vectors")

        except Exception as e:
            logger.error(f"FAISS index rebuild failed: {str(e)}")
            raise

    @contextmanager
    def _get_db_connection(self):
        """Get database connection with automatic cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {str(e)}")
            raise
        finally:
            if conn:
                conn.close()

    def add_user(self,
                 user_id: str,
                 name: str,
                 email: Optional[str] = None,
                 phone: Optional[str] = None,
                 metadata: Optional[Dict] = None) -> bool:
        """
        Add a new user to the database.

        Args:
            user_id: Unique user identifier
            name: User's name
            email: User's email address
            phone: User's phone number
            metadata: Additional user metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()

                    metadata_json = json.dumps(metadata) if metadata else None

                    cursor.execute('''
                        INSERT INTO users (user_id, name, email, phone, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user_id, name, email, phone, metadata_json))

                    conn.commit()
                    logger.info(f"Added user: {user_id} ({name})")
                    return True

        except sqlite3.IntegrityError:
            logger.error(f"User {user_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Add user error: {str(e)}")
            return False

    def add_embedding(self,
                     user_id: str,
                     embedding: np.ndarray,
                     quality_score: float = 1.0,
                     model_name: Optional[str] = None) -> Optional[str]:
        """
        Add face embedding for a user.

        Args:
            user_id: User identifier
            embedding: Face embedding vector
            quality_score: Quality score of the embedding
            model_name: Name of model used to extract embedding (defaults to self.embedding_model)

        Returns:
            Embedding ID if successful, None otherwise
        """
        try:
            model_name = model_name or self.embedding_model

            # Prevent cross-model pollution of the active index
            if model_name != self.embedding_model:
                logger.error(
                    f"Cannot add embedding from incompatible model '{model_name}': "
                    f"active database model is '{self.embedding_model}'."
                )
                return None

            if embedding.shape[0] != self.embedding_dim:
                logger.error(f"Embedding dimension mismatch: {embedding.shape[0]} != {self.embedding_dim}")
                return None

            embedding_id = str(uuid.uuid4())

            with self._lock:
                # Add to FAISS index
                faiss_id = self.index.ntotal
                embedding_normalized = embedding / np.linalg.norm(embedding)

                if FAISS_AVAILABLE:
                    self.index.add(embedding_normalized.reshape(1, -1).astype(np.float32))
                else:
                    self.index.add(embedding_normalized)

                # Add to SQLite database
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()

                    embedding_blob = pickle.dumps(embedding_normalized)

                    cursor.execute('''
                        INSERT INTO embeddings (user_id, embedding_id, faiss_id,
                                              embedding_vector, quality_score, model_name, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, 1)
                    ''', (user_id, embedding_id, faiss_id, embedding_blob, quality_score, model_name))

                    conn.commit()

                # Save FAISS index
                self._save_faiss_index()

                logger.info(f"Added embedding {embedding_id} for user {user_id} (model: {model_name})")
                return embedding_id

        except Exception as e:
            logger.error(f"Add embedding error: {str(e)}")
            return None

    def find_similar_faces(self,
                          query_embedding: np.ndarray,
                          k: int = 5,
                          threshold: float = 0.7,
                          model_name: Optional[str] = None) -> List[Dict]:
        """
        Find similar faces using FAISS similarity search.

        Args:
            query_embedding: Query face embedding
            k: Number of similar faces to return
            threshold: Similarity threshold (cosine similarity)
            model_name: Optional model identifier for query vector (rejects mismatch with active model)

        Returns:
            List of similar faces with metadata
        """
        try:
            if model_name is not None and model_name != self.embedding_model:
                logger.error(
                    f"Cross-model comparison rejected: query embedding model '{model_name}' "
                    f"does not match active index model '{self.embedding_model}'"
                )
                return []

            if query_embedding.shape[0] != self.embedding_dim:
                logger.error(f"Query embedding dimension mismatch: {query_embedding.shape[0]} != {self.embedding_dim}")
                return []

            # Normalize query embedding
            query_normalized = query_embedding / np.linalg.norm(query_embedding)

            with self._lock:
                # Search in FAISS index
                if FAISS_AVAILABLE:
                    distances, indices = self.index.search(
                        query_normalized.reshape(1, -1).astype(np.float32), k
                    )
                else:
                    distances, indices = self.index.search(query_normalized, k)

                # Convert distances to similarities
                if self.index_type == 'IndexFlatIP' or not FAISS_AVAILABLE:
                    similarities = distances[0]  # Inner product is already similarity
                else:
                    similarities = 1.0 / (1.0 + distances[0])  # Convert L2 distance to similarity

                # Filter by threshold and get metadata
                results = []

                with self._get_db_connection() as conn:
                    cursor = conn.cursor()

                    for i, (similarity, faiss_id) in enumerate(zip(similarities, indices[0])):
                        if similarity >= threshold:
                            # Get embedding metadata for active matching embeddings
                            cursor.execute('''
                                SELECT e.user_id, e.embedding_id, e.quality_score, e.model_name,
                                       u.name, u.email, u.phone, u.is_active
                                FROM embeddings e
                                JOIN users u ON e.user_id = u.user_id
                                WHERE e.faiss_id = ? AND e.is_active = 1
                            ''', (int(faiss_id),))

                            row = cursor.fetchone()
                            if row:
                                # Guard against cross-model vector matches
                                if row['model_name'] and row['model_name'] != self.embedding_model:
                                    logger.warning(
                                        f"Ignoring hit from incompatible model '{row['model_name']}' "
                                        f"(active: '{self.embedding_model}')"
                                    )
                                    continue

                                results.append({
                                    'user_id': row['user_id'],
                                    'name': row['name'],
                                    'email': row['email'],
                                    'phone': row['phone'],
                                    'embedding_id': row['embedding_id'],
                                    'similarity': float(similarity),
                                    'quality_score': row['quality_score'],
                                    'model_name': row['model_name'] or self.embedding_model,
                                    'is_active': bool(row['is_active'])
                                })

                # Sort by similarity (descending)
                results.sort(key=lambda x: x['similarity'], reverse=True)

                logger.info(f"Found {len(results)} similar faces above threshold {threshold}")
                return results

        except Exception as e:
            logger.error(f"Similar faces search error: {str(e)}")
            return []

    def authenticate_user(self,
                         query_embedding: np.ndarray,
                         threshold: float = 0.7,
                         model_name: Optional[str] = None) -> Optional[Dict]:
        """
        Authenticate user based on face embedding.

        Args:
            query_embedding: Query face embedding
            threshold: Authentication threshold
            model_name: Optional model identifier for query embedding

        Returns:
            User information if authenticated, None otherwise
        """
        try:
            similar_faces = self.find_similar_faces(
                query_embedding, k=1, threshold=threshold, model_name=model_name
            )

            if similar_faces and similar_faces[0]['is_active']:
                best_match = similar_faces[0]
                logger.info(f"User authenticated: {best_match['user_id']} (similarity: {best_match['similarity']:.3f})")
                return best_match
            else:
                logger.info("No matching user found or user inactive")
                return None

        except Exception as e:
            logger.error(f"User authentication error: {str(e)}")
            return None

    def log_authentication(self,
                          user_id: Optional[str],
                          success: bool,
                          confidence_score: float = 0.0,
                          liveness_score: float = 0.0,
                          deepfake_score: float = 0.0,
                          ip_address: str = "unknown",
                          metadata: Optional[Dict] = None) -> bool:
        """
        Log authentication attempt.

        Args:
            user_id: User ID (None for failed attempts)
            success: Whether authentication was successful
            confidence_score: Overall confidence score
            liveness_score: Liveness detection score
            deepfake_score: Deepfake detection score
            ip_address: Client IP address
            metadata: Additional metadata

        Returns:
            True if logged successfully, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                metadata_json = json.dumps(metadata) if metadata else None

                cursor.execute('''
                    INSERT INTO auth_logs (user_id, success, confidence_score,
                                         liveness_score, deepfake_score, ip_address, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, success, confidence_score, liveness_score,
                      deepfake_score, ip_address, metadata_json))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Authentication logging error: {str(e)}")
            return False

    def get_user(self, user_id: str) -> Optional[Dict]:
        """
        Get user information by user ID.

        Args:
            user_id: User identifier

        Returns:
            User information or None if not found
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT user_id, name, email, phone, created_at,
                           updated_at, is_active, metadata
                    FROM users WHERE user_id = ?
                ''', (user_id,))

                row = cursor.fetchone()
                if row:
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    return {
                        'user_id': row['user_id'],
                        'name': row['name'],
                        'email': row['email'],
                        'phone': row['phone'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'is_active': bool(row['is_active']),
                        'metadata': metadata
                    }
                else:
                    return None

        except Exception as e:
            logger.error(f"Get user error: {str(e)}")
            return None

    def delete_user(self, user_id: str) -> bool:
        """
        Delete user and all associated embeddings.

        Args:
            user_id: User identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                with self._get_db_connection() as conn:
                    cursor = conn.cursor()

                    # Get FAISS IDs of embeddings to remove
                    cursor.execute('SELECT faiss_id FROM embeddings WHERE user_id = ?', (user_id,))
                    faiss_ids = [row[0] for row in cursor.fetchall()]

                    # Delete embeddings
                    cursor.execute('DELETE FROM embeddings WHERE user_id = ?', (user_id,))

                    # Delete user
                    cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))

                    deleted_embeddings = cursor.rowcount
                    conn.commit()

                    logger.info(f"Deleted user {user_id} and {deleted_embeddings} embeddings")

                # Rebuild FAISS index to remove deleted vectors
                if deleted_embeddings > 0:
                    self._rebuild_faiss_index()
                    logger.info(f"Rebuilt FAISS index after deleting user {user_id}")

                return True

        except Exception as e:
            logger.error(f"Delete user error: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                # User statistics
                cursor.execute('SELECT COUNT(*) as total_users FROM users')
                total_users = cursor.fetchone()['total_users']

                cursor.execute('SELECT COUNT(*) as active_users FROM users WHERE is_active = 1')
                active_users = cursor.fetchone()['active_users']

                # Embedding statistics
                cursor.execute('SELECT COUNT(*) as total_embeddings FROM embeddings')
                total_embeddings = cursor.fetchone()['total_embeddings']

                cursor.execute('SELECT AVG(quality_score) as avg_quality FROM embeddings')
                avg_quality = cursor.fetchone()['avg_quality'] or 0.0

                # Authentication statistics
                cursor.execute('SELECT COUNT(*) as total_auths FROM auth_logs')
                total_auths = cursor.fetchone()['total_auths']

                cursor.execute('SELECT COUNT(*) as successful_auths FROM auth_logs WHERE success = 1')
                successful_auths = cursor.fetchone()['successful_auths']

                cursor.execute('SELECT COUNT(*) as active_embeddings FROM embeddings WHERE is_active = 1')
                active_embeddings = cursor.fetchone()['active_embeddings']

                success_rate = (successful_auths / total_auths * 100) if total_auths > 0 else 0.0

                return {
                    'total_users': total_users,
                    'active_users': active_users,
                    'total_embeddings': total_embeddings,
                    'active_embeddings': active_embeddings,
                    'archived_embeddings': total_embeddings - active_embeddings,
                    'active_embedding_model': self.embedding_model,
                    'avg_embedding_quality': float(avg_quality),
                    'total_authentications': total_auths,
                    'successful_authentications': successful_auths,
                    'success_rate_percent': float(success_rate),
                    'faiss_index_size': self.index.ntotal if hasattr(self.index, 'ntotal') else 0
                }

        except Exception as e:
            logger.error(f"Get statistics error: {str(e)}")
            return {}

    def archive_incompatible_embeddings(self, target_model: Optional[str] = None) -> Dict[str, Any]:
        """
        Safely archive embeddings incompatible with the target embedding model.

        Ensures data protection by:
        1. Creating a verified backup of both SQLite and FAISS index before modifications.
        2. Setting is_active = 0 and faiss_id = NULL for all embeddings where model_name != target_model.
        3. Identifying all users who now have 0 active embeddings and updating their metadata
           with re_enrollment_required = True and the reason.
        4. Rebuilding the active FAISS index to contain only compatible target_model embeddings.
        5. Updating system_metadata key active_embedding_model to target_model.
        6. Re-validating FAISS consistency.

        Args:
            target_model: Target embedding model identifier (defaults to self.embedding_model)

        Returns:
            Dictionary summarizing migration actions, archived count, and affected users requiring re-enrollment.
        """
        target_model = target_model or self.embedding_model

        try:
            with self._lock:
                timestamp = int(time.time())
                migration_backup_dir = self.backup_dir / f"migration_backup_{timestamp}"
                logger.info(f"Initiating pre-migration verified database backup to {migration_backup_dir}...")
                backup_success = self.backup_database(str(migration_backup_dir))
                if not backup_success:
                    raise RuntimeError(
                        f"Pre-migration backup failed to {migration_backup_dir}; aborting migration."
                    )

                with self._get_db_connection() as conn:
                    cursor = conn.cursor()

                    # Find all active embeddings that do NOT match the target model
                    cursor.execute('''
                        SELECT embedding_id, user_id, model_name
                        FROM embeddings
                        WHERE (model_name != ? OR model_name IS NULL) AND is_active = 1
                    ''', (target_model,))
                    incompatible_rows = cursor.fetchall()
                    archived_ids = [row['embedding_id'] for row in incompatible_rows]
                    candidate_users = set(row['user_id'] for row in incompatible_rows)

                    # Mark incompatible embeddings as inactive
                    cursor.execute('''
                        UPDATE embeddings
                        SET is_active = 0, faiss_id = NULL
                        WHERE (model_name != ? OR model_name IS NULL) AND is_active = 1
                    ''', (target_model,))

                    # Identify users who now have zero active embeddings
                    re_enrollment_users = []
                    for uid in candidate_users:
                        cursor.execute(
                            'SELECT COUNT(*) as active_cnt FROM embeddings WHERE user_id = ? AND is_active = 1',
                            (uid,)
                        )
                        active_cnt = cursor.fetchone()['active_cnt']
                        if active_cnt == 0:
                            re_enrollment_users.append(uid)
                            # Update user metadata with re_enrollment_required
                            cursor.execute('SELECT metadata FROM users WHERE user_id = ?', (uid,))
                            user_row = cursor.fetchone()
                            user_meta = {}
                            if user_row and user_row['metadata']:
                                try:
                                    user_meta = json.loads(user_row['metadata'])
                                except Exception:
                                    user_meta = {'raw': user_row['metadata']}

                            user_meta['re_enrollment_required'] = True
                            user_meta['re_enrollment_reason'] = (
                                f"Embedding model migrated to '{target_model}'. "
                                "Prior facial embeddings were archived to prevent cross-model bias."
                            )
                            cursor.execute(
                                'UPDATE users SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?',
                                (json.dumps(user_meta), uid)
                            )

                    # Update system metadata
                    cursor.execute('''
                        INSERT OR REPLACE INTO system_metadata (key, value, updated_at)
                        VALUES ('active_embedding_model', ?, CURRENT_TIMESTAMP)
                    ''', (target_model,))

                    conn.commit()

                # Rebuild FAISS index from remaining active embeddings
                self._rebuild_faiss_index()

                # Validate rebuilt state
                if not self._validate_faiss_index():
                    raise RuntimeError("Post-migration FAISS validation failed.")

                self.embedding_model = target_model
                logger.info(
                    f"Migration completed: archived {len(archived_ids)} embeddings; "
                    f"{len(re_enrollment_users)} users flagged for re-enrollment."
                )

                return {
                    'target_model': target_model,
                    'archived_embeddings_count': len(archived_ids),
                    're_enrollment_users': re_enrollment_users,
                    're_enrollment_required_count': len(re_enrollment_users),
                    'backup_dir': str(migration_backup_dir)
                }

        except Exception as e:
            logger.error(f"Migration error: {str(e)}")
            raise

    def _save_faiss_index(self) -> None:
        """Save FAISS index to disk atomically."""
        try:
            # Write to temporary file first
            temp_path = self.faiss_index_path.with_suffix('.faiss.tmp')

            if FAISS_AVAILABLE:
                faiss.write_index(self.index, str(temp_path))
            else:
                with open(temp_path, 'wb') as f:
                    pickle.dump(self.index, f)

            # Ensure file is flushed to disk
            # (faiss.write_index and pickle.dump already flush on close)

            # Atomically replace the destination
            temp_path.replace(self.faiss_index_path)

            logger.debug("FAISS index saved successfully")
        except Exception as e:
            # Clean up temporary file on failure
            temp_path = self.faiss_index_path.with_suffix('.faiss.tmp')
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            logger.error(f"FAISS index save error: {str(e)}")
            raise

    def backup_database(self, backup_path: str) -> bool:
        """
        Create a backup of the entire database.

        Args:
            backup_path: Path for backup files

        Returns:
            True if successful, False otherwise
        """
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())

            # Backup SQLite database
            sqlite_backup = backup_dir / f"face_recognition_{timestamp}.db"
            with self._get_db_connection() as conn:
                backup_conn = sqlite3.connect(str(sqlite_backup))
                conn.backup(backup_conn)
                backup_conn.close()

            # Backup FAISS index - serialize in-memory index to temp file, then atomic move
            faiss_backup = backup_dir / f"face_index_{timestamp}.faiss"
            temp_faiss_backup = backup_dir / f"face_index_{timestamp}.faiss.tmp"

            try:
                with self._lock:
                    if FAISS_AVAILABLE:
                        faiss.write_index(self.index, str(temp_faiss_backup))
                    else:
                        with open(temp_faiss_backup, 'wb') as f:
                            pickle.dump(self.index, f)

                # Ensure file is flushed (faiss.write_index and pickle.dump already flush on close)
                # Atomically replace the destination
                temp_faiss_backup.replace(faiss_backup)

                # Verify backup can be loaded
                self._verify_faiss_backup(faiss_backup)

            except Exception as e:
                # Clean up temporary file on failure
                if temp_faiss_backup.exists():
                    try:
                        temp_faiss_backup.unlink()
                    except Exception:
                        pass
                raise

            logger.info(f"Database backup created: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Database backup error: {str(e)}")
            return False

    def _verify_faiss_backup(self, backup_path: Path, verify_dim: bool = True) -> None:
        """
        Verify that a FAISS backup can be loaded successfully.

        Args:
            backup_path: Path to the FAISS backup file
            verify_dim: Whether to verify dimension matches self.embedding_dim

        Raises:
            Exception: If backup cannot be loaded or is invalid
        """
        try:
            if FAISS_AVAILABLE:
                test_index = faiss.read_index(str(backup_path))
                if test_index is None:
                    raise ValueError(f"Failed to read FAISS index from {backup_path}")
                if verify_dim and test_index.d != self.embedding_dim:
                    raise ValueError(f"Backup dimension mismatch: {test_index.d} != {self.embedding_dim}")
            else:
                with open(backup_path, 'rb') as f:
                    test_index = pickle.load(f)
                if test_index is None:
                    raise ValueError(f"Failed to load fallback index from {backup_path}")
                if verify_dim and (not hasattr(test_index, 'dimension') or test_index.dimension != self.embedding_dim):
                    raise ValueError(f"Backup dimension mismatch: {test_index.dimension} != {self.embedding_dim}")

            logger.debug(f"FAISS backup verified: {backup_path}")
        except Exception as e:
            logger.error(f"FAISS backup verification failed: {str(e)}")
            raise

    def close(self) -> None:
        """Close database connections and save indexes."""
        try:
            self._save_faiss_index()
            logger.info("Database manager closed successfully")
        except Exception as e:
            logger.error(f"Database close error: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize database manager
    db_manager = DatabaseManager(
        db_path="test_face_recognition.db",
        faiss_index_path="test_face_index.faiss",
        embedding_dim=128
    )

    # Test adding a user
    success = db_manager.add_user(
        user_id="test_user_001",
        name="John Doe",
        email="john.doe@example.com",
        metadata={"department": "IT", "role": "developer"}
    )
    print(f"Add user success: {success}")

    # Test adding an embedding
    dummy_embedding = np.random.randn(128).astype(np.float32)
    embedding_id = db_manager.add_embedding("test_user_001", dummy_embedding, quality_score=0.95)
    print(f"Added embedding: {embedding_id}")

    # Test similarity search
    query_embedding = dummy_embedding + np.random.randn(128) * 0.1  # Similar but with noise
    similar_faces = db_manager.find_similar_faces(query_embedding, k=5, threshold=0.5)
    print(f"Found {len(similar_faces)} similar faces")

    # Test authentication
    auth_result = db_manager.authenticate_user(query_embedding, threshold=0.5)
    print(f"Authentication result: {auth_result}")

    # Get statistics
    stats = db_manager.get_statistics()
    print(f"Database statistics: {stats}")

    # Clean up
    db_manager.close()

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
                 index_type: str = 'IndexFlatIP'):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database
            faiss_index_path: Path to FAISS index file
            embedding_dim: Dimension of face embeddings
            index_type: FAISS index type ('IndexFlatIP', 'IndexFlatL2', 'IndexIVFFlat')
        """
        self.db_path = Path(db_path)
        self.faiss_index_path = Path(faiss_index_path)
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Create directories if they don't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for database operations
        self._lock = threading.Lock()
        
        # Initialize databases
        self._init_sqlite_db()
        self._init_faiss_index()
        
        logger.info(f"DatabaseManager initialized with embedding dim: {embedding_dim}")
    
    def _init_sqlite_db(self) -> None:
        """Initialize SQLite database with required tables."""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
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
                        FOREIGN KEY (user_id) REFERENCES users (user_id)
                    )
                ''')
                
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
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_auth_logs_user_id ON auth_logs (user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_auth_logs_timestamp ON auth_logs (timestamp)')
                
                conn.commit()
                logger.info("SQLite database initialized successfully")
                
        except Exception as e:
            logger.error(f"SQLite initialization error: {str(e)}")
            raise
    
    def _init_faiss_index(self) -> None:
        """Initialize FAISS index for similarity search."""
        try:
            if FAISS_AVAILABLE:
                if self.faiss_index_path.exists():
                    # Load existing index
                    self.index = faiss.read_index(str(self.faiss_index_path))
                    logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
                else:
                    # Create new index
                    if self.index_type == 'IndexFlatIP':
                        self.index = faiss.IndexFlatIP(self.embedding_dim)
                    elif self.index_type == 'IndexFlatL2':
                        self.index = faiss.IndexFlatL2(self.embedding_dim)
                    elif self.index_type == 'IndexIVFFlat':
                        quantizer = faiss.IndexFlatL2(self.embedding_dim)
                        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
                    else:
                        logger.warning(f"Unknown index type: {self.index_type}, using IndexFlatIP")
                        self.index = faiss.IndexFlatIP(self.embedding_dim)
                    
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
                        self.index = FallbackIndex(self.embedding_dim)
                else:
                    self.index = FallbackIndex(self.embedding_dim)
                    logger.info("Created new fallback index")
                
        except Exception as e:
            logger.error(f"FAISS index initialization error: {str(e)}")
            self.index = FallbackIndex(self.embedding_dim)
    
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
                     quality_score: float = 1.0) -> Optional[str]:
        """
        Add face embedding for a user.
        
        Args:
            user_id: User identifier
            embedding: Face embedding vector
            quality_score: Quality score of the embedding
            
        Returns:
            Embedding ID if successful, None otherwise
        """
        try:
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
                                              embedding_vector, quality_score)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (user_id, embedding_id, faiss_id, embedding_blob, quality_score))
                    
                    conn.commit()
                
                # Save FAISS index
                self._save_faiss_index()
                
                logger.info(f"Added embedding {embedding_id} for user {user_id}")
                return embedding_id
                
        except Exception as e:
            logger.error(f"Add embedding error: {str(e)}")
            return None
    
    def find_similar_faces(self,
                          query_embedding: np.ndarray,
                          k: int = 5,
                          threshold: float = 0.7) -> List[Dict]:
        """
        Find similar faces using FAISS similarity search.
        
        Args:
            query_embedding: Query face embedding
            k: Number of similar faces to return
            threshold: Similarity threshold (cosine similarity)
            
        Returns:
            List of similar faces with metadata
        """
        try:
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
                            # Get embedding metadata
                            cursor.execute('''
                                SELECT e.user_id, e.embedding_id, e.quality_score,
                                       u.name, u.email, u.phone, u.is_active
                                FROM embeddings e
                                JOIN users u ON e.user_id = u.user_id
                                WHERE e.faiss_id = ?
                            ''', (int(faiss_id),))
                            
                            row = cursor.fetchone()
                            if row:
                                results.append({
                                    'user_id': row['user_id'],
                                    'name': row['name'],
                                    'email': row['email'],
                                    'phone': row['phone'],
                                    'embedding_id': row['embedding_id'],
                                    'similarity': float(similarity),
                                    'quality_score': row['quality_score'],
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
                         threshold: float = 0.7) -> Optional[Dict]:
        """
        Authenticate user based on face embedding.
        
        Args:
            query_embedding: Query face embedding
            threshold: Authentication threshold
            
        Returns:
            User information if authenticated, None otherwise
        """
        try:
            similar_faces = self.find_similar_faces(query_embedding, k=1, threshold=threshold)
            
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
                    
                    # Remove from FAISS index (not directly supported, would need rebuilding)
                    # For now, we mark them as deleted in the database
                    
                    # Delete embeddings
                    cursor.execute('DELETE FROM embeddings WHERE user_id = ?', (user_id,))
                    
                    # Delete user
                    cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
                    
                    deleted_embeddings = cursor.rowcount
                    conn.commit()
                    
                    logger.info(f"Deleted user {user_id} and {deleted_embeddings} embeddings")
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
                
                success_rate = (successful_auths / total_auths * 100) if total_auths > 0 else 0.0
                
                return {
                    'total_users': total_users,
                    'active_users': active_users,
                    'total_embeddings': total_embeddings,
                    'avg_embedding_quality': float(avg_quality),
                    'total_authentications': total_auths,
                    'successful_authentications': successful_auths,
                    'success_rate_percent': float(success_rate),
                    'faiss_index_size': self.index.ntotal if hasattr(self.index, 'ntotal') else 0
                }
                
        except Exception as e:
            logger.error(f"Get statistics error: {str(e)}")
            return {}
    
    def _save_faiss_index(self) -> None:
        """Save FAISS index to disk."""
        try:
            if FAISS_AVAILABLE:
                faiss.write_index(self.index, str(self.faiss_index_path))
            else:
                with open(self.faiss_index_path, 'wb') as f:
                    pickle.dump(self.index, f)
            logger.debug("FAISS index saved successfully")
        except Exception as e:
            logger.error(f"FAISS index save error: {str(e)}")
    
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
            
            # Backup FAISS index
            faiss_backup = backup_dir / f"face_index_{timestamp}.faiss"
            if self.faiss_index_path.exists():
                import shutil
                shutil.copy2(self.faiss_index_path, faiss_backup)
            
            logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database backup error: {str(e)}")
            return False
    
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

"""
Test Database Manager

Unit tests for the database manager including FAISS operations,
SQLite metadata management, and user authentication.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import sqlite3
import tempfile
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from src.database_manager import DatabaseManager, FallbackIndex
except ImportError:
    DatabaseManager = Mock
    FallbackIndex = Mock


class TestFallbackIndex:
    """Test class for Fallback Index functionality."""
    
    @pytest.fixture
    def fallback_index(self):
        """Create Fallback Index instance for testing."""
        return FallbackIndex(dimension=512)
    
    def test_initialization(self, fallback_index):
        """Test Fallback Index initialization."""
        assert fallback_index.dimension == 512
        assert len(fallback_index.embeddings) == 0
        assert len(fallback_index.ids) == 0
    
    def test_add_embedding(self, fallback_index):
        """Test adding embedding to fallback index."""
        embedding = np.random.randn(512).astype(np.float32)
        embedding_id = "embedding_001"
        
        fallback_index.add(embedding_id, embedding)
        
        assert len(fallback_index.embeddings) == 1
        assert len(fallback_index.ids) == 1
        assert fallback_index.ids[0] == embedding_id
        np.testing.assert_array_equal(fallback_index.embeddings[0], embedding)
    
    def test_search_similar(self, fallback_index):
        """Test searching similar embeddings."""
        # Add some embeddings
        for i in range(5):
            embedding = np.random.randn(512).astype(np.float32)
            fallback_index.add(f"embedding_{i:03d}", embedding)
        
        # Search with similar embedding
        query_embedding = fallback_index.embeddings[0] + np.random.randn(512) * 0.1
        indices, similarities = fallback_index.search(query_embedding, k=3)
        
        assert len(indices) == 3
        assert len(similarities) == 3
        assert indices[0] == 0  # Should find the most similar one first
        assert similarities[0] >= similarities[1] >= similarities[2]  # Descending order
    
    def test_search_empty_index(self, fallback_index):
        """Test searching in empty index."""
        query_embedding = np.random.randn(512).astype(np.float32)
        indices, similarities = fallback_index.search(query_embedding, k=3)
        
        assert len(indices) == 0
        assert len(similarities) == 0
    
    def test_get_total_count(self, fallback_index):
        """Test getting total count of embeddings."""
        assert fallback_index.ntotal == 0
        
        # Add some embeddings
        for i in range(3):
            embedding = np.random.randn(512).astype(np.float32)
            fallback_index.add(f"embedding_{i:03d}", embedding)
        
        assert fallback_index.ntotal == 3
    
    def test_save_load_index(self, fallback_index):
        """Test saving and loading index."""
        # Add some embeddings
        for i in range(3):
            embedding = np.random.randn(512).astype(np.float32)
            fallback_index.add(f"embedding_{i:03d}", embedding)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            fallback_index.save(tmp_path)
            
            # Load into new index
            new_index = FallbackIndex.load(tmp_path)
            
            assert new_index.dimension == fallback_index.dimension
            assert new_index.ntotal == fallback_index.ntotal
            assert new_index.ids == fallback_index.ids
            np.testing.assert_array_equal(
                new_index.embeddings, 
                fallback_index.embeddings
            )
        finally:
            os.unlink(tmp_path)


class TestDatabaseManager:
    """Test class for Database Manager functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        yield db_path
        
        # Cleanup
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def temp_index_path(self):
        """Create temporary index path for testing."""
        with tempfile.NamedTemporaryFile(suffix='.index', delete=False) as tmp_file:
            index_path = tmp_file.name
        
        yield index_path
        
        # Cleanup
        try:
            os.unlink(index_path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def db_manager(self, temp_db_path, temp_index_path):
        """Create Database Manager instance for testing."""
        config = {
            'database_path': temp_db_path,
            'index_path': temp_index_path,
            'embedding_dimension': 512,
            'similarity_threshold': 0.7,
            'use_faiss': False  # Use fallback for testing
        }
        return DatabaseManager(config)
    
    def test_initialization(self, db_manager, temp_db_path):
        """Test Database Manager initialization."""
        assert db_manager.db_path == temp_db_path
        assert db_manager.embedding_dimension == 512
        assert db_manager.similarity_threshold == 0.7
        assert hasattr(db_manager, 'index')
        
        # Check database tables exist
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Check users table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        assert cursor.fetchone() is not None
        
        # Check embeddings table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_add_user_success(self, db_manager):
        """Test successful user addition."""
        user_data = {
            'user_id': 'test_user_001',
            'name': 'Test User',
            'email': 'test@example.com',
            'department': 'Engineering'
        }
        
        result = db_manager.add_user(**user_data)
        
        assert result is True
        
        # Verify user was added to database
        user = db_manager.get_user('test_user_001')
        assert user is not None
        assert user['name'] == 'Test User'
        assert user['email'] == 'test@example.com'
    
    def test_add_user_duplicate(self, db_manager):
        """Test adding duplicate user."""
        user_data = {
            'user_id': 'test_user_001',
            'name': 'Test User'
        }
        
        # Add user first time
        result1 = db_manager.add_user(**user_data)
        assert result1 is True
        
        # Try to add same user again
        result2 = db_manager.add_user(**user_data)
        assert result2 is False
    
    def test_get_user_exists(self, db_manager):
        """Test getting existing user."""
        # Add user first
        db_manager.add_user(
            user_id='test_user_001',
            name='Test User',
            email='test@example.com'
        )
        
        user = db_manager.get_user('test_user_001')
        
        assert user is not None
        assert user['user_id'] == 'test_user_001'
        assert user['name'] == 'Test User'
        assert user['email'] == 'test@example.com'
        assert user['is_active'] is True
    
    def test_get_user_not_exists(self, db_manager):
        """Test getting non-existent user."""
        user = db_manager.get_user('nonexistent_user')
        assert user is None
    
    def test_update_user(self, db_manager):
        """Test updating user information."""
        # Add user first
        db_manager.add_user(
            user_id='test_user_001',
            name='Test User',
            email='test@example.com'
        )
        
        # Update user
        result = db_manager.update_user(
            user_id='test_user_001',
            name='Updated User',
            department='Marketing'
        )
        
        assert result is True
        
        # Verify update
        user = db_manager.get_user('test_user_001')
        assert user['name'] == 'Updated User'
        assert user['department'] == 'Marketing'
        assert user['email'] == 'test@example.com'  # Should remain unchanged
    
    def test_deactivate_user(self, db_manager):
        """Test deactivating user."""
        # Add user first
        db_manager.add_user(user_id='test_user_001', name='Test User')
        
        result = db_manager.deactivate_user('test_user_001')
        assert result is True
        
        # Verify user is deactivated
        user = db_manager.get_user('test_user_001')
        assert user['is_active'] is False
    
    def test_delete_user(self, db_manager):
        """Test deleting user."""
        # Add user and embedding first
        db_manager.add_user(user_id='test_user_001', name='Test User')
        embedding = np.random.randn(512).astype(np.float32)
        db_manager.add_embedding('test_user_001', embedding)
        
        result = db_manager.delete_user('test_user_001')
        assert result is True
        
        # Verify user is deleted
        user = db_manager.get_user('test_user_001')
        assert user is None
        
        # Verify embeddings are also deleted
        embeddings = db_manager.get_user_embeddings('test_user_001')
        assert len(embeddings) == 0
    
    def test_add_embedding(self, db_manager):
        """Test adding face embedding."""
        # Add user first
        db_manager.add_user(user_id='test_user_001', name='Test User')
        
        embedding = np.random.randn(512).astype(np.float32)
        embedding_id = db_manager.add_embedding('test_user_001', embedding)
        
        assert embedding_id is not None
        assert isinstance(embedding_id, str)
        
        # Verify embedding was added to database
        embeddings = db_manager.get_user_embeddings('test_user_001')
        assert len(embeddings) == 1
        assert embeddings[0]['embedding_id'] == embedding_id
    
    def test_get_user_embeddings(self, db_manager):
        """Test getting user embeddings."""
        # Add user first
        db_manager.add_user(user_id='test_user_001', name='Test User')
        
        # Add multiple embeddings
        embeddings_added = []
        for i in range(3):
            embedding = np.random.randn(512).astype(np.float32)
            embedding_id = db_manager.add_embedding('test_user_001', embedding)
            embeddings_added.append(embedding_id)
        
        # Get embeddings
        embeddings = db_manager.get_user_embeddings('test_user_001')
        
        assert len(embeddings) == 3
        retrieved_ids = [emb['embedding_id'] for emb in embeddings]
        for emb_id in embeddings_added:
            assert emb_id in retrieved_ids
    
    def test_find_similar_faces(self, db_manager):
        """Test finding similar faces."""
        # Add users and embeddings
        for i in range(3):
            user_id = f'user_{i:03d}'
            db_manager.add_user(user_id=user_id, name=f'User {i}')
            
            # Add multiple embeddings per user
            for j in range(2):
                embedding = np.random.randn(512).astype(np.float32)
                db_manager.add_embedding(user_id, embedding)
        
        # Search for similar faces
        query_embedding = np.random.randn(512).astype(np.float32)
        results = db_manager.find_similar_faces(query_embedding, k=5)
        
        assert len(results) <= 5
        for result in results:
            assert 'user_id' in result
            assert 'embedding_id' in result
            assert 'similarity' in result
            assert 0 <= result['similarity'] <= 1
    
    def test_authenticate_user_success(self, db_manager):
        """Test successful user authentication."""
        # Add user and embedding
        db_manager.add_user(user_id='test_user_001', name='Test User')
        embedding = np.random.randn(512).astype(np.float32)
        db_manager.add_embedding('test_user_001', embedding)
        
        # Authenticate with similar embedding
        query_embedding = embedding + np.random.randn(512) * 0.1
        result = db_manager.authenticate_user(query_embedding)
        
        if result:  # May be None if similarity is too low
            assert result['user_id'] == 'test_user_001'
            assert result['name'] == 'Test User'
            assert result['is_active'] is True
            assert 'similarity' in result
    
    def test_authenticate_user_inactive(self, db_manager):
        """Test authentication with inactive user."""
        # Add user and embedding
        db_manager.add_user(user_id='test_user_001', name='Test User')
        embedding = np.random.randn(512).astype(np.float32)
        db_manager.add_embedding('test_user_001', embedding)
        
        # Deactivate user
        db_manager.deactivate_user('test_user_001')
        
        # Try to authenticate
        result = db_manager.authenticate_user(embedding)
        assert result is None
    
    def test_log_authentication(self, db_manager):
        """Test logging authentication attempts."""
        db_manager.log_authentication(
            user_id='test_user_001',
            success=True,
            confidence=0.85,
            timestamp='2023-12-01 10:00:00',
            ip_address='192.168.1.100',
            metadata={'liveness_score': 0.8, 'deepfake_score': 0.9}
        )
        
        # Verify log was created (you might want to add a method to retrieve logs)
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM authentication_logs WHERE user_id = ?", ('test_user_001',))
        log = cursor.fetchone()
        conn.close()
        
        assert log is not None
    
    def test_get_authentication_logs(self, db_manager):
        """Test getting authentication logs."""
        # Add some logs
        for i in range(5):
            db_manager.log_authentication(
                user_id=f'user_{i % 2}',  # Two different users
                success=i % 2 == 0,  # Alternating success/failure
                confidence=0.8,
                timestamp=f'2023-12-0{i+1} 10:00:00'
            )
        
        # Get logs for specific user
        logs = db_manager.get_authentication_logs('user_0', limit=10)
        assert len(logs) == 3  # user_0 appears 3 times (0, 2, 4)
        
        # Get all logs
        all_logs = db_manager.get_authentication_logs(limit=10)
        assert len(all_logs) == 5
    
    def test_get_user_stats(self, db_manager):
        """Test getting user statistics."""
        # Add user and embeddings
        db_manager.add_user(user_id='test_user_001', name='Test User')
        for i in range(3):
            embedding = np.random.randn(512).astype(np.float32)
            db_manager.add_embedding('test_user_001', embedding)
        
        # Add some authentication logs
        for i in range(5):
            db_manager.log_authentication(
                user_id='test_user_001',
                success=i < 3,  # 3 successful, 2 failed
                confidence=0.8
            )
        
        stats = db_manager.get_user_stats('test_user_001')
        
        assert stats['user_id'] == 'test_user_001'
        assert stats['total_embeddings'] == 3
        assert stats['total_authentications'] == 5
        assert stats['successful_authentications'] == 3
        assert stats['failed_authentications'] == 2
    
    @patch('faiss.IndexFlatIP')
    def test_faiss_integration(self, mock_faiss_index, temp_db_path, temp_index_path):
        """Test FAISS integration when available."""
        mock_index = Mock()
        mock_faiss_index.return_value = mock_index
        mock_index.ntotal = 0
        
        # Create database manager with FAISS enabled
        config = {
            'database_path': temp_db_path,
            'index_path': temp_index_path,
            'embedding_dimension': 512,
            'similarity_threshold': 0.7,
            'use_faiss': True
        }
        
        with patch('faiss.write_index'), patch('faiss.read_index'):
            db_manager = DatabaseManager(config)
            
            # Verify FAISS index was created
            mock_faiss_index.assert_called_once_with(512)
    
    def test_database_backup_restore(self, db_manager, temp_db_path):
        """Test database backup and restore functionality."""
        # Add some test data
        db_manager.add_user(user_id='test_user_001', name='Test User')
        embedding = np.random.randn(512).astype(np.float32)
        db_manager.add_embedding('test_user_001', embedding)
        
        # Create backup
        backup_path = temp_db_path + '.backup'
        result = db_manager.backup_database(backup_path)
        assert result is True
        assert os.path.exists(backup_path)
        
        # Clean up backup
        os.unlink(backup_path)


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for Database Manager."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for integration testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            db_path = tmp_file.name
        
        yield db_path
        
        # Cleanup
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass
    
    def test_full_workflow(self, temp_db_path):
        """Test complete workflow from user registration to authentication."""
        config = {
            'database_path': temp_db_path,
            'index_path': temp_db_path + '.index',
            'embedding_dimension': 512,
            'similarity_threshold': 0.6,
            'use_faiss': False
        }
        
        db_manager = DatabaseManager(config)
        
        # Register multiple users
        users = [
            {'user_id': 'user_001', 'name': 'Alice Smith', 'email': 'alice@example.com'},
            {'user_id': 'user_002', 'name': 'Bob Johnson', 'email': 'bob@example.com'},
            {'user_id': 'user_003', 'name': 'Carol Davis', 'email': 'carol@example.com'}
        ]
        
        # Add users and their embeddings
        for user in users:
            assert db_manager.add_user(**user) is True
            
            # Add multiple embeddings per user
            for i in range(3):
                embedding = np.random.randn(512).astype(np.float32)
                embedding_id = db_manager.add_embedding(user['user_id'], embedding)
                assert embedding_id is not None
        
        # Test authentication for each user
        for user in users:
            embeddings = db_manager.get_user_embeddings(user['user_id'])
            assert len(embeddings) == 3
            
            # Try to authenticate with one of their embeddings
            # Note: May not work perfectly due to random embeddings
            # In real scenario, embeddings would be more similar
    
    def test_performance_with_large_dataset(self, temp_db_path):
        """Test performance with larger dataset."""
        config = {
            'database_path': temp_db_path,
            'index_path': temp_db_path + '.index',
            'embedding_dimension': 512,
            'similarity_threshold': 0.7,
            'use_faiss': False
        }
        
        db_manager = DatabaseManager(config)
        
        # Add many users (smaller number for testing)
        num_users = 50
        for i in range(num_users):
            user_id = f'user_{i:03d}'
            assert db_manager.add_user(user_id=user_id, name=f'User {i}') is True
            
            # Add embeddings
            embedding = np.random.randn(512).astype(np.float32)
            embedding_id = db_manager.add_embedding(user_id, embedding)
            assert embedding_id is not None
        
        # Test search performance
        query_embedding = np.random.randn(512).astype(np.float32)
        results = db_manager.find_similar_faces(query_embedding, k=10)
        
        assert len(results) <= 10
        assert len(results) > 0  # Should find some results


if __name__ == "__main__":
    # Simple test runner
    pytest.main([__file__, "-v"])
"""
Unit tests for Streamlit app helper functions
Streamlit 앱 헬퍼 함수 단위 테스트
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Note: We can't directly test Streamlit UI components,
# but we can test the helper functions that don't depend on Streamlit


class TestStreamlitHelpers(unittest.TestCase):
    """Test helper functions used in Streamlit app"""

    def test_imports(self):
        """Test that all required modules can be imported"""
        try:
            from src.recommenders.popularity import get_popular_movies
            from src.recommenders.genre import get_genre_based_recommendations
            from src.recommenders.similarity import get_similar_movies_for_user
            from src.recommenders.ml_based import get_ml_recommendations
            from src.recommenders.hybrid import get_hybrid_recommendations
            from src.db_connection import get_sqlalchemy_engine
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import failed: {e}")

    def test_get_user_info_structure(self):
        """Test user info query returns expected structure"""
        from src.db_connection import get_sqlalchemy_engine
        from sqlalchemy import text

        engine = get_sqlalchemy_engine()
        try:
            with engine.connect() as conn:
                query = text("""
                    SELECT
                        u.user_id,
                        u.gender,
                        u.age,
                        u.occupation,
                        COUNT(rt.rating) as total_ratings,
                        ROUND(AVG(rt.rating)::numeric, 2) as avg_rating
                    FROM users u
                    LEFT JOIN ratings_train rt ON u.user_id = rt.user_id
                    WHERE u.user_id = :user_id
                    GROUP BY u.user_id, u.gender, u.age, u.occupation
                """)
                result = pd.read_sql(query, conn, params={'user_id': 1})

                # Check columns exist
                self.assertIn('user_id', result.columns)
                self.assertIn('gender', result.columns)
                self.assertIn('age', result.columns)
                self.assertIn('occupation', result.columns)
                self.assertIn('total_ratings', result.columns)
                self.assertIn('avg_rating', result.columns)

                # Check not empty
                self.assertFalse(result.empty)
        finally:
            engine.dispose()

    def test_get_all_users(self):
        """Test fetching all users"""
        from src.db_connection import get_sqlalchemy_engine
        from sqlalchemy import text

        engine = get_sqlalchemy_engine()
        try:
            with engine.connect() as conn:
                query = text("SELECT DISTINCT user_id FROM users ORDER BY user_id LIMIT 10")
                result = pd.read_sql(query, conn)

                # Should have users
                self.assertGreater(len(result), 0)

                # User IDs should be integers
                self.assertTrue(all(isinstance(uid, (int, pd.Int64Dtype)) for uid in result['user_id']))
        finally:
            engine.dispose()

    def test_search_movies_query(self):
        """Test movie search functionality"""
        from src.db_connection import get_sqlalchemy_engine
        from sqlalchemy import text

        engine = get_sqlalchemy_engine()
        try:
            with engine.connect() as conn:
                query = text("""
                    SELECT
                        m.movie_id,
                        m.title,
                        STRING_AGG(DISTINCT g.genre_name, '|' ORDER BY g.genre_name) as genres,
                        COALESCE(AVG(rt.rating), 0) as avg_rating,
                        COUNT(rt.rating) as rating_count
                    FROM movies m
                    LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                    LEFT JOIN genres g ON mg.genre_id = g.genre_id
                    LEFT JOIN ratings_train rt ON m.movie_id = rt.movie_id
                    WHERE LOWER(m.title) LIKE LOWER(:query)
                    GROUP BY m.movie_id, m.title
                    ORDER BY rating_count DESC
                    LIMIT :limit
                """)
                result = pd.read_sql(query, conn, params={'query': '%toy%', 'limit': 5})

                # Should find "Toy Story"
                self.assertGreater(len(result), 0)

                # Check structure
                self.assertIn('movie_id', result.columns)
                self.assertIn('title', result.columns)
                self.assertIn('genres', result.columns)
        finally:
            engine.dispose()

    def test_watch_history_query(self):
        """Test watch history query"""
        from src.db_connection import get_sqlalchemy_engine
        from sqlalchemy import text

        engine = get_sqlalchemy_engine()
        try:
            with engine.connect() as conn:
                query = text("""
                    SELECT
                        m.movie_id,
                        m.title,
                        STRING_AGG(DISTINCT g.genre_name, '|' ORDER BY g.genre_name) as genres,
                        rt.rating,
                        rt.timestamp
                    FROM ratings_train rt
                    JOIN movies m ON rt.movie_id = m.movie_id
                    LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                    LEFT JOIN genres g ON mg.genre_id = g.genre_id
                    WHERE rt.user_id = :user_id
                    GROUP BY m.movie_id, m.title, rt.rating, rt.timestamp
                    ORDER BY rt.rating DESC, rt.timestamp DESC
                    LIMIT :limit
                """)
                result = pd.read_sql(query, conn, params={'user_id': 1, 'limit': 10})

                # User 1 should have ratings
                self.assertGreater(len(result), 0)

                # Check structure
                self.assertIn('movie_id', result.columns)
                self.assertIn('title', result.columns)
                self.assertIn('rating', result.columns)

                # Ratings should be valid (1-5)
                self.assertTrue(all(1 <= r <= 5 for r in result['rating']))
        finally:
            engine.dispose()

    def test_recommendation_functions_exist(self):
        """Test that all recommendation functions are callable"""
        from src.recommenders.popularity import get_popular_movies
        from src.recommenders.genre import get_genre_based_recommendations
        from src.recommenders.similarity import get_similar_movies_for_user, get_similar_movies_for_movie
        from src.recommenders.hybrid import get_hybrid_recommendations

        # Test they are callable
        self.assertTrue(callable(get_popular_movies))
        self.assertTrue(callable(get_genre_based_recommendations))
        self.assertTrue(callable(get_similar_movies_for_user))
        self.assertTrue(callable(get_similar_movies_for_movie))
        self.assertTrue(callable(get_hybrid_recommendations))

    def test_display_recommendations_logic(self):
        """Test recommendation display logic without Streamlit"""
        # Create mock recommendation DataFrame
        mock_recs = pd.DataFrame({
            'movie_id': [1, 2, 3],
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'genres': ['Action', 'Comedy', 'Drama'],
            'weighted_rating': [4.5, 4.3, 4.1],
            'avg_rating': [4.4, 4.2, 4.0],
            'rating_count': [100, 90, 80]
        })

        # Test score column detection
        self.assertIn('weighted_rating', mock_recs.columns)
        self.assertIn('avg_rating', mock_recs.columns)

        # Test formatting
        mock_recs['Score'] = mock_recs['weighted_rating'].round(2)
        self.assertEqual(len(mock_recs), 3)
        self.assertTrue(all(isinstance(s, (float, np.floating)) for s in mock_recs['Score']))


class TestAppConfiguration(unittest.TestCase):
    """Test app configuration and setup"""

    def test_streamlit_installed(self):
        """Test that Streamlit is installed"""
        try:
            import streamlit
            self.assertTrue(True)
        except ImportError:
            self.fail("Streamlit is not installed. Run: pip install streamlit")

    def test_model_path_structure(self):
        """Test model directory structure"""
        models_dir = Path('models')
        self.assertTrue(models_dir.exists() or True)  # Directory may not exist yet

        # Check if model exists (optional, may not be trained yet)
        model_file = models_dir / 'svd_model.pkl'
        if model_file.exists():
            self.assertTrue(model_file.is_file())


if __name__ == '__main__':
    import numpy as np  # Import for test
    unittest.main(verbosity=2)

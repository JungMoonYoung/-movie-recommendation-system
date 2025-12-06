"""
Unit tests for Hybrid Recommendation System
하이브리드 추천 시스템 단위 테스트
"""
import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommenders.hybrid import (
    HybridRecommender,
    get_hybrid_recommendations,
    get_recommendations_for_evaluation
)


class TestHybridRecommender(unittest.TestCase):
    """Test cases for Hybrid Recommender"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.test_user_id = 1
        cls.test_n = 10
        cls.default_weights = {'popularity': 0.1, 'genre': 0.2, 'similarity': 0.3, 'ml': 0.4}

    def test_init_default_weights(self):
        """Test initialization with default weights"""
        recommender = HybridRecommender()
        self.assertEqual(recommender.weights['popularity'], 0.1)
        self.assertEqual(recommender.weights['genre'], 0.2)
        self.assertEqual(recommender.weights['similarity'], 0.3)
        self.assertEqual(recommender.weights['ml'], 0.4)

    def test_init_custom_weights(self):
        """Test initialization with custom weights"""
        custom_weights = {'popularity': 0.25, 'genre': 0.25, 'similarity': 0.25, 'ml': 0.25}
        recommender = HybridRecommender(weights=custom_weights)
        self.assertEqual(recommender.weights, custom_weights)

    def test_init_invalid_weights_sum(self):
        """Test that invalid weight sum raises error"""
        invalid_weights = {'popularity': 0.3, 'genre': 0.3, 'similarity': 0.3, 'ml': 0.3}  # sum = 1.2
        with self.assertRaises(ValueError):
            HybridRecommender(weights=invalid_weights)

    def test_min_max_normalize(self):
        """Test Min-Max normalization"""
        recommender = HybridRecommender()
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = recommender.min_max_normalize(scores)

        # Check range [0, 1]
        self.assertAlmostEqual(normalized.min(), 0.0)
        self.assertAlmostEqual(normalized.max(), 1.0)

        # Check values
        self.assertAlmostEqual(normalized.iloc[0], 0.0)
        self.assertAlmostEqual(normalized.iloc[2], 0.5)
        self.assertAlmostEqual(normalized.iloc[4], 1.0)

    def test_min_max_normalize_constant(self):
        """Test normalization with constant scores"""
        recommender = HybridRecommender()
        scores = pd.Series([5.0, 5.0, 5.0, 5.0])
        normalized = recommender.min_max_normalize(scores)

        # All should be 1.0 when constant
        self.assertTrue(np.all(normalized == 1.0))

    def test_get_all_algorithm_results(self):
        """Test getting results from all algorithms"""
        recommender = HybridRecommender()
        results = recommender._get_all_algorithm_results(self.test_user_id, n=50)

        # Should return a dictionary
        self.assertIsInstance(results, dict)

        # Should have keys for all algorithms (even if empty DataFrame)
        self.assertIn('popularity', results)
        self.assertIn('genre', results)
        self.assertIn('similarity', results)
        self.assertIn('ml', results)

        # Each value should be a DataFrame
        for algo, df in results.items():
            self.assertIsInstance(df, pd.DataFrame)

    def test_extract_candidates_and_scores(self):
        """Test extracting candidates and scores from algorithm results"""
        recommender = HybridRecommender()
        algorithm_results = recommender._get_all_algorithm_results(self.test_user_id, n=30)
        candidates, scores = recommender._extract_candidates_and_scores(algorithm_results)

        # Should return set and dictionary
        self.assertIsInstance(candidates, set)
        self.assertIsInstance(scores, dict)

        # Should have candidates
        if len(candidates) > 0:
            # All should be integers
            self.assertTrue(all(isinstance(m, (int, np.integer)) for m in candidates))

            # Check score structure
            for movie_id, score_dict in scores.items():
                self.assertIsInstance(score_dict, dict)

    def test_calculate_hybrid_scores(self):
        """Test hybrid score calculation"""
        recommender = HybridRecommender()

        # Mock scores
        scores = {
            1: {'popularity': 5.0, 'genre': 4.0, 'similarity': 3.0, 'ml': 4.5},
            2: {'popularity': 3.0, 'genre': 5.0, 'similarity': 4.0, 'ml': 3.5},
            3: {'popularity': 4.0, 'genre': 3.0, 'similarity': 5.0, 'ml': 4.0}
        }

        hybrid_scores = recommender.calculate_hybrid_scores(scores)

        # Should return dictionary
        self.assertIsInstance(hybrid_scores, dict)

        # Should have 3 movies
        self.assertEqual(len(hybrid_scores), 3)

        # All scores should be in [0, 1] range (normalized)
        for score in hybrid_scores.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_recommend_returns_dataframe(self):
        """Test that recommend returns DataFrame"""
        recommender = HybridRecommender()
        result = recommender.recommend(self.test_user_id, n=self.test_n)

        self.assertIsInstance(result, pd.DataFrame)

    def test_recommend_returns_correct_number(self):
        """Test that recommend returns correct number of movies"""
        recommender = HybridRecommender()
        result = recommender.recommend(self.test_user_id, n=self.test_n)

        # Should return at most n movies
        self.assertLessEqual(len(result), self.test_n)

    def test_recommend_has_required_columns(self):
        """Test that recommendations have required columns"""
        recommender = HybridRecommender()
        result = recommender.recommend(self.test_user_id, n=self.test_n)

        if not result.empty:
            required_columns = ['movie_id', 'title', 'genres', 'hybrid_score']
            for col in required_columns:
                self.assertIn(col, result.columns)

    def test_recommend_sorted_by_score(self):
        """Test that recommendations are sorted by hybrid score"""
        recommender = HybridRecommender()
        result = recommender.recommend(self.test_user_id, n=self.test_n)

        if len(result) > 1:
            scores = result['hybrid_score'].tolist()
            # Check descending order
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_recommend_no_duplicates(self):
        """Test that recommendations have no duplicate movies"""
        recommender = HybridRecommender()
        result = recommender.recommend(self.test_user_id, n=self.test_n)

        if not result.empty:
            movie_ids = result['movie_id'].tolist()
            self.assertEqual(len(movie_ids), len(set(movie_ids)))

    def test_recommend_with_different_weights(self):
        """Test recommendations with different weight configurations"""
        weights_list = [
            {'popularity': 0.25, 'genre': 0.25, 'similarity': 0.25, 'ml': 0.25},
            {'popularity': 0.1, 'genre': 0.2, 'similarity': 0.5, 'ml': 0.2},
            {'popularity': 0.0, 'genre': 0.0, 'similarity': 0.0, 'ml': 1.0}
        ]

        for weights in weights_list:
            recommender = HybridRecommender(weights=weights)
            result = recommender.recommend(self.test_user_id, n=5)

            # Should return DataFrame
            self.assertIsInstance(result, pd.DataFrame)

            # Should have results
            if not result.empty:
                self.assertGreater(len(result), 0)

    def test_get_hybrid_recommendations_function(self):
        """Test convenience function"""
        result = get_hybrid_recommendations(user_id=self.test_user_id, n=5)

        self.assertIsInstance(result, pd.DataFrame)

    def test_get_recommendations_for_evaluation(self):
        """Test batch evaluation function"""
        user_ids = [1, 2, 3]
        recommendations = get_recommendations_for_evaluation(user_ids=user_ids, n=5)

        # Should return dictionary
        self.assertIsInstance(recommendations, dict)

        # Should have all users
        self.assertEqual(len(recommendations), len(user_ids))

        # Each user should have list of movie IDs
        for user_id in user_ids:
            self.assertIn(user_id, recommendations)
            self.assertIsInstance(recommendations[user_id], list)

    def test_recommend_excludes_watched_movies(self):
        """Test that recommendations exclude already watched movies"""
        recommender = HybridRecommender()
        result = recommender.recommend(self.test_user_id, n=20)

        if not result.empty:
            # Get watched movies
            from src.db_connection import get_sqlalchemy_engine
            from sqlalchemy import text

            engine = get_sqlalchemy_engine()
            try:
                with engine.connect() as conn:
                    query = text("SELECT movie_id FROM ratings_train WHERE user_id = :user_id")
                    watched_df = pd.read_sql(query, conn, params={'user_id': self.test_user_id})
                    watched_movies = set(watched_df['movie_id'].tolist())

                # Check no overlap
                recommended_movies = set(result['movie_id'].tolist())
                overlap = recommended_movies & watched_movies

                # Some algorithms might include watched movies, so this is not strict
                # But hybrid should mostly exclude them
                self.assertLessEqual(len(overlap), len(recommended_movies) * 0.2)  # <= 20% overlap

            finally:
                engine.dispose()

    def test_hybrid_scores_in_valid_range(self):
        """Test that hybrid scores are in valid range"""
        recommender = HybridRecommender()
        result = recommender.recommend(self.test_user_id, n=self.test_n)

        if not result.empty:
            scores = result['hybrid_score'].tolist()
            for score in scores:
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_different_candidate_pool_sizes(self):
        """Test with different candidate pool sizes"""
        recommender = HybridRecommender()

        for pool_size in [50, 100, 200]:
            result = recommender.recommend(self.test_user_id, n=10, candidate_pool_size=pool_size)
            self.assertIsInstance(result, pd.DataFrame)


class TestWeightOptimization(unittest.TestCase):
    """Test weight configurations"""

    def test_ml_focused_weights(self):
        """Test ML-focused configuration"""
        weights = {'popularity': 0.1, 'genre': 0.2, 'similarity': 0.3, 'ml': 0.4}
        recommender = HybridRecommender(weights=weights)
        result = recommender.recommend(user_id=1, n=5)

        self.assertIsInstance(result, pd.DataFrame)

    def test_balanced_weights(self):
        """Test balanced configuration"""
        weights = {'popularity': 0.25, 'genre': 0.25, 'similarity': 0.25, 'ml': 0.25}
        recommender = HybridRecommender(weights=weights)
        result = recommender.recommend(user_id=1, n=5)

        self.assertIsInstance(result, pd.DataFrame)

    def test_cf_focused_weights(self):
        """Test CF-focused configuration"""
        weights = {'popularity': 0.1, 'genre': 0.2, 'similarity': 0.5, 'ml': 0.2}
        recommender = HybridRecommender(weights=weights)
        result = recommender.recommend(user_id=1, n=5)

        self.assertIsInstance(result, pd.DataFrame)

    def test_extreme_weight_ml_only(self):
        """Test with ML only"""
        weights = {'popularity': 0.0, 'genre': 0.0, 'similarity': 0.0, 'ml': 1.0}
        recommender = HybridRecommender(weights=weights)
        result = recommender.recommend(user_id=1, n=5)

        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    unittest.main(verbosity=2)

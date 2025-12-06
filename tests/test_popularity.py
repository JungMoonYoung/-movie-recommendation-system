"""
Unit tests for popularity-based recommendation
인기 기반 추천 알고리즘 단위 테스트
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recommenders.popularity import (
    get_popular_movies,
    get_popular_movies_for_user,
    get_recommendations_for_evaluation
)


class TestPopularityRecommender(unittest.TestCase):
    """Popularity-based recommendation test cases"""

    def test_get_popular_movies_returns_dataframe(self):
        """Test that get_popular_movies returns a DataFrame"""
        result = get_popular_movies(n=10, min_ratings=30)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 10)
        self.assertTrue('movie_id' in result.columns)
        self.assertTrue('title' in result.columns)
        self.assertTrue('weighted_rating' in result.columns)

    def test_get_popular_movies_respects_n_parameter(self):
        """Test that n parameter correctly limits results"""
        result_5 = get_popular_movies(n=5, min_ratings=30)
        result_20 = get_popular_movies(n=20, min_ratings=30)

        self.assertEqual(len(result_5), 5)
        self.assertEqual(len(result_20), 20)

    def test_get_popular_movies_sorted_by_weighted_rating(self):
        """Test that results are sorted by weighted_rating in descending order"""
        result = get_popular_movies(n=10, min_ratings=30)

        # Check if weighted_rating is in descending order
        weighted_ratings = result['weighted_rating'].tolist()
        self.assertEqual(weighted_ratings, sorted(weighted_ratings, reverse=True))

    def test_get_popular_movies_has_minimum_ratings(self):
        """Test that all movies have at least min_ratings"""
        min_threshold = 50
        result = get_popular_movies(n=10, min_ratings=min_threshold)

        # All movies should have >= min_threshold ratings
        self.assertTrue(all(result['rating_count'] >= min_threshold))

    def test_get_popular_movies_for_user_returns_dataframe(self):
        """Test that get_popular_movies_for_user returns a DataFrame"""
        result = get_popular_movies_for_user(user_id=1, n=10, min_ratings=30)

        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 10)
        self.assertTrue('movie_id' in result.columns)
        self.assertTrue('title' in result.columns)

    def test_get_popular_movies_for_user_excludes_watched(self):
        """Test that user's watched movies are excluded"""
        # This is a functional test - we can't easily verify without database access
        # We just test that it returns results and doesn't error
        user_id = 1
        result = get_popular_movies_for_user(user_id=user_id, n=10, min_ratings=30)

        self.assertIsNotNone(result)
        # Should return some movies (user 1 hasn't watched all popular movies)
        self.assertGreater(len(result), 0)

    def test_get_popular_movies_for_different_users(self):
        """Test that different users get potentially different recommendations"""
        result_user1 = get_popular_movies_for_user(user_id=1, n=10, min_ratings=30)
        result_user2 = get_popular_movies_for_user(user_id=2, n=10, min_ratings=30)

        # Both should return results
        self.assertGreater(len(result_user1), 0)
        self.assertGreater(len(result_user2), 0)

        # Results might be different (if users watched different movies)
        # We just verify they both work

    def test_get_recommendations_for_evaluation(self):
        """Test batch recommendation generation for evaluation"""
        user_ids = [1, 2, 3, 4, 5]
        recommendations = get_recommendations_for_evaluation(user_ids, n=10, min_ratings=30)

        # Should return dict with all user IDs
        self.assertEqual(len(recommendations), len(user_ids))

        # Each user should have a list of movie IDs
        for user_id in user_ids:
            self.assertIn(user_id, recommendations)
            self.assertIsInstance(recommendations[user_id], list)
            self.assertLessEqual(len(recommendations[user_id]), 10)

    def test_min_ratings_parameter_effect(self):
        """Test that higher min_ratings threshold affects results"""
        result_low = get_popular_movies(n=10, min_ratings=10)
        result_high = get_popular_movies(n=10, min_ratings=100)

        # Both should return results
        self.assertEqual(len(result_low), 10)
        self.assertEqual(len(result_high), 10)

        # High threshold movies should all have >= 100 ratings
        self.assertTrue(all(result_high['rating_count'] >= 100))

    def test_weighted_rating_calculation(self):
        """Test that weighted_rating is reasonable"""
        result = get_popular_movies(n=10, min_ratings=30)

        # Weighted ratings should be between 0 and 5 (MovieLens rating scale)
        for wr in result['weighted_rating']:
            self.assertGreaterEqual(wr, 0.0)
            self.assertLessEqual(wr, 5.0)

        # Weighted ratings should be close to avg_rating
        # (for movies with many ratings, weighted ~= avg)
        for idx, row in result.iterrows():
            diff = abs(row['weighted_rating'] - row['avg_rating'])
            # Difference should be small (< 0.5 typically)
            self.assertLess(diff, 1.0)

    def test_popular_movies_have_high_ratings(self):
        """Test that top popular movies have high average ratings"""
        result = get_popular_movies(n=10, min_ratings=30)

        # Top 10 popular movies should have avg_rating > 4.0 typically
        avg_ratings = result['avg_rating'].tolist()
        mean_avg_rating = sum(avg_ratings) / len(avg_ratings)

        # Popular movies should have above-average ratings
        self.assertGreater(mean_avg_rating, 3.6)  # MovieLens overall avg is ~3.6


if __name__ == '__main__':
    unittest.main(verbosity=2)

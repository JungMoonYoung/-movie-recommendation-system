"""
Unit tests for genre-based recommendation
장르 기반 추천 알고리즘 단위 테스트
"""
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recommenders.genre import (
    get_user_genre_preference,
    get_genre_based_recommendations,
    get_recommendations_for_evaluation
)


class TestGenreRecommender(unittest.TestCase):
    """Genre-based recommendation test cases"""

    def test_get_user_genre_preference_returns_dataframe(self):
        """Test that get_user_genre_preference returns a DataFrame"""
        result = get_user_genre_preference(user_id=1, min_ratings_per_genre=3, top_n=5)

        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 5)
        self.assertTrue('genre_name' in result.columns)
        self.assertTrue('rating_count' in result.columns)
        self.assertTrue('avg_rating' in result.columns)
        self.assertTrue('preference_score' in result.columns)

    def test_get_user_genre_preference_sorted_by_score(self):
        """Test that results are sorted by preference_score"""
        result = get_user_genre_preference(user_id=1, min_ratings_per_genre=3, top_n=5)

        if len(result) > 1:
            scores = result['preference_score'].tolist()
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_get_user_genre_preference_min_ratings_filter(self):
        """Test that all genres meet minimum rating count"""
        min_threshold = 5
        result = get_user_genre_preference(user_id=1, min_ratings_per_genre=min_threshold, top_n=5)

        if len(result) > 0:
            self.assertTrue(all(result['rating_count'] >= min_threshold))

    def test_get_genre_based_recommendations_returns_dataframe(self):
        """Test that get_genre_based_recommendations returns a DataFrame"""
        result = get_genre_based_recommendations(user_id=1, n=10)

        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 10)
        self.assertTrue('movie_id' in result.columns)
        self.assertTrue('title' in result.columns)
        self.assertTrue('preferred_genre' in result.columns)
        self.assertTrue('recommendation_score' in result.columns)

    def test_genre_recommendations_sorted_by_score(self):
        """Test that recommendations are sorted by recommendation_score"""
        result = get_genre_based_recommendations(user_id=1, n=10)

        if len(result) > 1:
            scores = result['recommendation_score'].tolist()
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_genre_recommendations_respects_n_parameter(self):
        """Test that n parameter correctly limits results"""
        result_5 = get_genre_based_recommendations(user_id=1, n=5)
        result_20 = get_genre_based_recommendations(user_id=1, n=20)

        self.assertLessEqual(len(result_5), 5)
        self.assertLessEqual(len(result_20), 20)

    def test_different_users_different_genres(self):
        """Test that different users may have different preferred genres"""
        pref_user1 = get_user_genre_preference(user_id=1, min_ratings_per_genre=3, top_n=3)
        pref_user2 = get_user_genre_preference(user_id=100, min_ratings_per_genre=3, top_n=3)

        # Both should return results
        self.assertGreater(len(pref_user1), 0)
        self.assertGreater(len(pref_user2), 0)

        # Top genres might be different (not guaranteed, but likely)
        # We just verify they both work

    def test_different_users_different_recommendations(self):
        """Test that different users get different recommendations"""
        rec_user1 = get_genre_based_recommendations(user_id=1, n=10)
        rec_user2 = get_genre_based_recommendations(user_id=100, n=10)

        # Both should return results
        self.assertGreater(len(rec_user1), 0)
        self.assertGreater(len(rec_user2), 0)

        # Movie lists should be different (personalized)
        if len(rec_user1) > 0 and len(rec_user2) > 0:
            movies_user1 = set(rec_user1['movie_id'].tolist())
            movies_user2 = set(rec_user2['movie_id'].tolist())
            # At least some movies should be different
            self.assertNotEqual(movies_user1, movies_user2)

    def test_get_recommendations_for_evaluation(self):
        """Test batch recommendation generation for evaluation"""
        user_ids = [1, 2, 3, 4, 5]
        recommendations = get_recommendations_for_evaluation(user_ids, n=10)

        # Should return dict with all user IDs
        self.assertEqual(len(recommendations), len(user_ids))

        # Each user should have a list of movie IDs
        for user_id in user_ids:
            self.assertIn(user_id, recommendations)
            self.assertIsInstance(recommendations[user_id], list)
            self.assertLessEqual(len(recommendations[user_id]), 10)

    def test_recommendation_score_calculation(self):
        """Test that recommendation_score is reasonable"""
        result = get_genre_based_recommendations(user_id=1, n=10)

        if len(result) > 0:
            # Scores should be positive
            for score in result['recommendation_score']:
                self.assertGreater(score, 0)

            # Scores should be influenced by both rating and genre preference
            # (specific value depends on data, just checking they exist)

    def test_genre_field_populated(self):
        """Test that preferred_genre field is populated"""
        result = get_genre_based_recommendations(user_id=1, n=10)

        if len(result) > 0:
            # All movies should have a preferred_genre
            self.assertFalse(result['preferred_genre'].isnull().any())

    def test_min_movie_ratings_parameter(self):
        """Test that min_movie_ratings parameter affects results"""
        result_low = get_genre_based_recommendations(user_id=1, n=10, min_movie_ratings=10)
        result_high = get_genre_based_recommendations(user_id=1, n=10, min_movie_ratings=50)

        # Both should return results (though potentially different counts)
        self.assertIsNotNone(result_low)
        self.assertIsNotNone(result_high)

        # Higher threshold might return fewer results
        if len(result_high) > 0:
            self.assertTrue(all(result_high['rating_count'] >= 50))

    def test_top_genres_parameter_effect(self):
        """Test that top_genres parameter affects recommendations"""
        result_1genre = get_genre_based_recommendations(user_id=1, n=10, top_genres=1)
        result_3genres = get_genre_based_recommendations(user_id=1, n=10, top_genres=3)

        # Both should work
        self.assertIsNotNone(result_1genre)
        self.assertIsNotNone(result_3genres)

        # More genres might provide more diverse recommendations
        # (not guaranteed, but both should return valid results)


if __name__ == '__main__':
    unittest.main(verbosity=2)

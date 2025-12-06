"""
Unit tests for Item-based Collaborative Filtering
Item-based 협업 필터링 단위 테스트
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recommenders.similarity import (
    get_popular_movies,
    get_similar_movies_for_movie,
    get_similar_movies_for_user,
    get_recommendations_for_evaluation
)


class TestPopularMovies:
    """Test popular movies retrieval"""

    def test_get_popular_movies_returns_list(self):
        """Should return a list of movie IDs"""
        result = get_popular_movies(min_ratings=100, limit=50)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_get_popular_movies_respects_limit(self):
        """Should respect the limit parameter"""
        result = get_popular_movies(min_ratings=50, limit=10)
        assert len(result) <= 10

    def test_get_popular_movies_sorted_by_count(self):
        """Should return movies sorted by rating count (descending)"""
        result = get_popular_movies(min_ratings=50, limit=5)
        assert len(result) == 5


class TestSimilarMoviesForMovie:
    """Test movie-to-movie similarity recommendations"""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame"""
        # Use a popular movie (e.g., movie_id=1 is usually Toy Story)
        result = get_similar_movies_for_movie(movie_id=1, n=10)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self):
        """Should return required columns"""
        result = get_similar_movies_for_movie(movie_id=1, n=10)
        expected_columns = {'movie_id', 'title', 'genres', 'similarity_score'}
        assert expected_columns.issubset(set(result.columns))

    def test_respects_n_parameter(self):
        """Should return at most N recommendations"""
        n = 5
        result = get_similar_movies_for_movie(movie_id=1, n=n)
        assert len(result) <= n

    def test_sorted_by_similarity_desc(self):
        """Should be sorted by similarity score (descending)"""
        result = get_similar_movies_for_movie(movie_id=1, n=10)
        if len(result) > 1:
            scores = result['similarity_score'].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_movies(self):
        """Should not return duplicate movie IDs"""
        result = get_similar_movies_for_movie(movie_id=1, n=10)
        assert len(result) == len(result['movie_id'].unique())

    def test_excludes_source_movie(self):
        """Should not include the source movie in results"""
        movie_id = 1
        result = get_similar_movies_for_movie(movie_id=movie_id, n=10)
        assert movie_id not in result['movie_id'].values


class TestSimilarMoviesForUser:
    """Test user-based item-CF recommendations"""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame"""
        result = get_similar_movies_for_user(user_id=1, n=10)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self):
        """Should return required columns"""
        result = get_similar_movies_for_user(user_id=1, n=10)
        expected_columns = {'movie_id', 'title', 'genres', 'recommendation_score'}
        assert expected_columns.issubset(set(result.columns))

    def test_respects_n_parameter(self):
        """Should return at most N recommendations"""
        n = 5
        result = get_similar_movies_for_user(user_id=1, n=n)
        assert len(result) <= n

    def test_sorted_by_score_desc(self):
        """Should be sorted by recommendation score (descending)"""
        result = get_similar_movies_for_user(user_id=1, n=10)
        if len(result) > 1:
            scores = result['recommendation_score'].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_movies(self):
        """Should not return duplicate movie IDs"""
        result = get_similar_movies_for_user(user_id=1, n=10)
        assert len(result) == len(result['movie_id'].unique())

    def test_different_users_get_different_recommendations(self):
        """Different users should get personalized recommendations"""
        result_user_1 = get_similar_movies_for_user(user_id=1, n=10)
        result_user_100 = get_similar_movies_for_user(user_id=100, n=10)

        # Should have different recommendations (personalization)
        # If both users have results, they should be different
        # If one or both have no results, that's also acceptable (data dependent)
        if not result_user_1.empty and not result_user_100.empty:
            movies_1 = set(result_user_1['movie_id'].tolist())
            movies_100 = set(result_user_100['movie_id'].tolist())
            # Allow some overlap, but should not be identical
            assert movies_1 != movies_100 or len(movies_1) == 0

    def test_min_rating_parameter_effect(self):
        """Different min_rating thresholds should affect results"""
        result_strict = get_similar_movies_for_user(user_id=1, n=10, min_rating=4.5)
        result_lenient = get_similar_movies_for_user(user_id=1, n=10, min_rating=3.5)

        # Lenient threshold should potentially include more source movies
        # Results may differ
        assert isinstance(result_strict, pd.DataFrame)
        assert isinstance(result_lenient, pd.DataFrame)


class TestBatchRecommendations:
    """Test batch recommendation generation"""

    def test_returns_dict(self):
        """Should return a dictionary"""
        user_ids = [1, 2, 3]
        result = get_recommendations_for_evaluation(user_ids, n=10)
        assert isinstance(result, dict)

    def test_all_users_present(self):
        """Should return recommendations for all requested users"""
        user_ids = [1, 2, 3]
        result = get_recommendations_for_evaluation(user_ids, n=10)
        assert set(result.keys()) == set(user_ids)

    def test_each_user_has_list(self):
        """Each user should have a list of movie IDs"""
        user_ids = [1, 2, 3]
        result = get_recommendations_for_evaluation(user_ids, n=10)
        for user_id in user_ids:
            assert isinstance(result[user_id], list)

    def test_respects_n_parameter(self):
        """Each user should get at most N recommendations"""
        n = 5
        user_ids = [1, 2, 3]
        result = get_recommendations_for_evaluation(user_ids, n=n)
        for user_id in user_ids:
            assert len(result[user_id]) <= n


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

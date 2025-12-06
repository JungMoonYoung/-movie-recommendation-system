"""
Hybrid Recommendation System
하이브리드 추천 시스템

Combines multiple recommendation algorithms with weighted scoring:
- Popularity-based (10%)
- Genre-based (20%)
- Item-based CF (30%)
- ML-based (40%)
"""
import numpy as np
import pandas as pd
import logging
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.recommenders.popularity import get_popular_movies
from src.recommenders.genre import get_genre_based_recommendations
from src.recommenders.similarity import get_similar_movies_for_user
from src.recommenders.ml_based import get_ml_recommendations
from src.db_connection import get_sqlalchemy_engine
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridRecommender:
    """
    Hybrid Recommender that combines multiple algorithms
    여러 알고리즘을 결합하는 하이브리드 추천 시스템
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        ml_model_path: str = 'models/svd_model.pkl'
    ):
        """
        Initialize Hybrid Recommender

        Args:
            weights: Algorithm weights (must sum to 1.0)
                     Default: {'popularity': 0.1, 'genre': 0.2, 'similarity': 0.3, 'ml': 0.4}
            ml_model_path: Path to trained ML model
        """
        if weights is None:
            self.weights = {
                'popularity': 0.1,
                'genre': 0.2,
                'similarity': 0.3,
                'ml': 0.4
            }
        else:
            # Validate weights sum to 1.0
            total = sum(weights.values())
            if not np.isclose(total, 1.0):
                raise ValueError(f"Weights must sum to 1.0, got {total}")
            self.weights = weights

        self.ml_model_path = ml_model_path

        logger.info(f"Hybrid Recommender initialized with weights: {self.weights}")

    def min_max_normalize(self, scores: pd.Series) -> pd.Series:
        """
        Apply Min-Max normalization to scores
        점수를 [0, 1] 범위로 정규화

        Args:
            scores: Series of scores

        Returns:
            pd.Series: Normalized scores in [0, 1] range
        """
        min_score = scores.min()
        max_score = scores.max()

        # Handle case where all scores are the same
        if np.isclose(min_score, max_score):
            return pd.Series(np.ones(len(scores)), index=scores.index)

        normalized = (scores - min_score) / (max_score - min_score)
        return normalized

    def _get_all_algorithm_results(self, user_id: int, n: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Get results from all algorithms in a single pass
        모든 알고리즘 결과를 한 번에 가져오기 (중복 제거)

        Args:
            user_id: User ID
            n: Number of recommendations per algorithm

        Returns:
            Dict[str, pd.DataFrame]: Algorithm name -> results DataFrame
        """
        results = {}

        # Popularity-based
        try:
            pop_recs = get_popular_movies(n=n, min_ratings=30)
            results['popularity'] = pop_recs if not pop_recs.empty else pd.DataFrame()
        except Exception as e:
            logger.warning(f"Failed to get popularity recommendations: {e}")
            results['popularity'] = pd.DataFrame()

        # Genre-based
        try:
            genre_recs = get_genre_based_recommendations(user_id=user_id, n=n, top_genres=3, min_movie_ratings=30)
            results['genre'] = genre_recs if not genre_recs.empty else pd.DataFrame()
        except Exception as e:
            logger.warning(f"Failed to get genre recommendations: {e}")
            results['genre'] = pd.DataFrame()

        # Item-based CF
        try:
            sim_recs = get_similar_movies_for_user(user_id=user_id, n=n, min_rating=4.0)
            results['similarity'] = sim_recs if not sim_recs.empty else pd.DataFrame()
        except Exception as e:
            logger.warning(f"Failed to get similarity recommendations: {e}")
            results['similarity'] = pd.DataFrame()

        # ML-based
        try:
            ml_recs = get_ml_recommendations(user_id=user_id, n=n, model_path=self.ml_model_path)
            results['ml'] = ml_recs if not ml_recs.empty else pd.DataFrame()
        except Exception as e:
            logger.warning(f"Failed to get ML recommendations: {e}")
            results['ml'] = pd.DataFrame()

        return results

    def _extract_candidates_and_scores(
        self,
        algorithm_results: Dict[str, pd.DataFrame]
    ) -> Tuple[set, Dict[int, Dict[str, float]]]:
        """
        Extract candidates and scores from algorithm results
        알고리즘 결과에서 후보와 점수 추출

        Args:
            algorithm_results: Results from all algorithms

        Returns:
            Tuple[set, Dict]: (candidate_movie_ids, scores_dict)
        """
        candidates = set()
        scores = {}

        # Extract candidates and scores from popularity
        if 'popularity' in algorithm_results and not algorithm_results['popularity'].empty:
            pop_df = algorithm_results['popularity']
            for idx, row in pop_df.iterrows():
                movie_id = row['movie_id']
                candidates.add(movie_id)
                if movie_id not in scores:
                    scores[movie_id] = {}
                scores[movie_id]['popularity'] = row.get('weighted_rating', 0.0)

        # Extract candidates and scores from genre
        if 'genre' in algorithm_results and not algorithm_results['genre'].empty:
            genre_df = algorithm_results['genre']
            for idx, row in genre_df.iterrows():
                movie_id = row['movie_id']
                candidates.add(movie_id)
                if movie_id not in scores:
                    scores[movie_id] = {}
                scores[movie_id]['genre'] = row.get('combined_score', 0.0)

        # Extract candidates and scores from similarity
        if 'similarity' in algorithm_results and not algorithm_results['similarity'].empty:
            sim_df = algorithm_results['similarity']
            for idx, row in sim_df.iterrows():
                movie_id = row['movie_id']
                candidates.add(movie_id)
                if movie_id not in scores:
                    scores[movie_id] = {}
                scores[movie_id]['similarity'] = row.get('recommendation_score', 0.0)

        # Extract candidates and scores from ML
        if 'ml' in algorithm_results and not algorithm_results['ml'].empty:
            ml_df = algorithm_results['ml']
            for idx, row in ml_df.iterrows():
                movie_id = row['movie_id']
                candidates.add(movie_id)
                if movie_id not in scores:
                    scores[movie_id] = {}
                scores[movie_id]['ml'] = row.get('predicted_rating', 0.0)

        logger.info(f"Collected {len(candidates)} candidate movies with scores")
        return candidates, scores

    def calculate_hybrid_scores(
        self,
        scores: Dict[int, Dict[str, float]]
    ) -> Dict[int, float]:
        """
        Calculate final hybrid scores with normalization
        정규화 후 최종 하이브리드 점수 계산

        Args:
            scores: {movie_id: {'popularity': score, 'genre': score, ...}}

        Returns:
            Dict[int, float]: {movie_id: hybrid_score}
        """
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(scores).T
        df = df.fillna(0.0)

        # Normalize each algorithm's scores to [0, 1]
        for algo in ['popularity', 'genre', 'similarity', 'ml']:
            if algo in df.columns and not df[algo].empty:
                df[f'{algo}_norm'] = self.min_max_normalize(df[algo])
            else:
                df[f'{algo}_norm'] = 0.0

        # Calculate weighted sum
        df['hybrid_score'] = 0.0
        for algo, weight in self.weights.items():
            norm_col = f'{algo}_norm'
            if norm_col in df.columns:
                df['hybrid_score'] += df[norm_col] * weight

        # Convert to dictionary
        hybrid_scores = df['hybrid_score'].to_dict()

        return hybrid_scores

    def recommend(self, user_id: int, n: int = 10, candidate_pool_size: int = 100) -> pd.DataFrame:
        """
        Generate hybrid recommendations for a user
        사용자를 위한 하이브리드 추천 생성

        Args:
            user_id: User ID
            n: Number of recommendations
            candidate_pool_size: Size of candidate pool per algorithm

        Returns:
            pd.DataFrame: Recommendations with hybrid scores
        """
        logger.info(f"Generating hybrid recommendations for user {user_id}...")
        start_time = time.time()

        # Step 1: Get all algorithm results in a single pass (OPTIMIZED - no redundant calls)
        algorithm_results = self._get_all_algorithm_results(user_id, n=candidate_pool_size)

        # Step 2: Extract candidates and scores from results (no re-computation needed)
        candidates, scores = self._extract_candidates_and_scores(algorithm_results)

        if len(candidates) == 0:
            logger.warning(f"No candidates found for user {user_id}")
            return pd.DataFrame()

        # Step 3: Calculate hybrid scores
        hybrid_scores = self.calculate_hybrid_scores(scores)

        # Step 4: Sort by hybrid score
        sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)

        # Step 5: Get top N
        top_movies = sorted_movies[:n]
        movie_ids = [m[0] for m in top_movies]
        scores_dict = {m[0]: m[1] for m in top_movies}

        # Step 6: Fetch movie details
        if len(movie_ids) == 0:
            return pd.DataFrame()

        engine = get_sqlalchemy_engine()
        try:
            with engine.connect() as conn:
                query = text("""
                    SELECT
                        m.movie_id,
                        m.title,
                        STRING_AGG(DISTINCT g.genre_name, '|' ORDER BY g.genre_name) as genres,
                        AVG(rt.rating) as avg_rating,
                        COUNT(rt.rating) as rating_count
                    FROM movies m
                    LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                    LEFT JOIN genres g ON mg.genre_id = g.genre_id
                    LEFT JOIN ratings_train rt ON m.movie_id = rt.movie_id
                    WHERE m.movie_id = ANY(:movie_ids)
                    GROUP BY m.movie_id, m.title
                """)

                result_df = pd.read_sql(query, conn, params={'movie_ids': movie_ids})

        except Exception as e:
            logger.error(f"Error fetching movie details: {e}")
            return pd.DataFrame()
        finally:
            engine.dispose()

        # Add hybrid scores
        result_df['hybrid_score'] = result_df['movie_id'].map(scores_dict)

        # Sort by hybrid score
        result_df = result_df.sort_values('hybrid_score', ascending=False)

        elapsed = time.time() - start_time
        logger.info(f"Hybrid recommendations completed in {elapsed:.3f} seconds")

        return result_df


def get_hybrid_recommendations(
    user_id: int,
    n: int = 10,
    weights: Dict[str, float] = None,
    ml_model_path: str = 'models/svd_model.pkl'
) -> pd.DataFrame:
    """
    Get hybrid recommendations for a user
    하이브리드 추천 생성 (편의 함수)

    Args:
        user_id: User ID
        n: Number of recommendations
        weights: Algorithm weights (default: {'popularity': 0.1, 'genre': 0.2, 'similarity': 0.3, 'ml': 0.4})
        ml_model_path: Path to trained ML model

    Returns:
        pd.DataFrame: Recommendations
    """
    recommender = HybridRecommender(weights=weights, ml_model_path=ml_model_path)
    return recommender.recommend(user_id, n=n)


def get_recommendations_for_evaluation(
    user_ids: List[int],
    n: int = 10,
    weights: Dict[str, float] = None,
    ml_model_path: str = 'models/svd_model.pkl'
) -> dict:
    """
    Get hybrid recommendations for multiple users (for evaluation)
    배치 평가용 하이브리드 추천 생성

    Args:
        user_ids: List of user IDs
        n: Number of recommendations per user
        weights: Algorithm weights
        ml_model_path: Path to trained ML model

    Returns:
        dict: {user_id: [movie_id_1, movie_id_2, ...]}
    """
    logger.info(f"Generating hybrid recommendations for {len(user_ids)} users...")

    recommender = HybridRecommender(weights=weights, ml_model_path=ml_model_path)
    recommendations = {}
    start_time = time.time()

    for idx, user_id in enumerate(user_ids):
        try:
            result_df = recommender.recommend(user_id, n=n)
            recommendations[user_id] = result_df['movie_id'].tolist()
        except Exception as e:
            logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
            recommendations[user_id] = []

        # Progress logging
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            users_per_sec = (idx + 1) / elapsed
            remaining = len(user_ids) - (idx + 1)
            eta = remaining / users_per_sec if users_per_sec > 0 else 0

            logger.info(
                f"Progress: {idx + 1}/{len(user_ids)} users "
                f"({(idx + 1) / len(user_ids) * 100:.1f}%) | "
                f"Speed: {users_per_sec:.1f} users/sec | "
                f"ETA: {eta / 60:.1f} min"
            )

    elapsed_time = time.time() - start_time
    logger.info(f"Batch evaluation completed in {elapsed_time / 60:.1f} minutes")

    return recommendations


if __name__ == "__main__":
    # Test hybrid recommendations
    logger.info("\n" + "=" * 60)
    logger.info("TESTING HYBRID RECOMMENDER")
    logger.info("=" * 60)

    # Test with different weight configurations
    weight_configs = [
        {
            'name': 'Default (ML-focused)',
            'weights': {'popularity': 0.1, 'genre': 0.2, 'similarity': 0.3, 'ml': 0.4}
        },
        {
            'name': 'Balanced',
            'weights': {'popularity': 0.25, 'genre': 0.25, 'similarity': 0.25, 'ml': 0.25}
        },
        {
            'name': 'CF-focused',
            'weights': {'popularity': 0.1, 'genre': 0.2, 'similarity': 0.5, 'ml': 0.2}
        }
    ]

    test_user_id = 1

    for config in weight_configs:
        logger.info(f"\n[Test] {config['name']}")
        logger.info(f"Weights: {config['weights']}")

        try:
            recs = get_hybrid_recommendations(
                user_id=test_user_id,
                n=10,
                weights=config['weights'],
                ml_model_path='models/svd_model.pkl'
            )

            print(f"\n{'=' * 60}")
            print(f"HYBRID RECOMMENDATIONS ({config['name']})")
            print(f"User: {test_user_id}")
            print("=" * 60)
            print(recs[['title', 'genres', 'hybrid_score', 'avg_rating', 'rating_count']].to_string(index=False))
            print("=" * 60)

        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)

    logger.info("\n[OK] Hybrid recommender test completed!")

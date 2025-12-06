"""
Popularity-based Recommendation
인기 기반 추천 알고리즘
"""
import pandas as pd
from sqlalchemy import text
import logging
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db_connection import get_sqlalchemy_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_popular_movies(n: int = 10, min_ratings: int = 30) -> pd.DataFrame:
    """
    전체 인기 영화 추천

    Bayesian weighted average를 사용하여 평점 수가 적은 영화의 과대평가 방지

    Args:
        n: 추천할 영화 개수
        min_ratings: 최소 평점 개수 (기본값: 30)

    Returns:
        pd.DataFrame: movie_id, title, release_year, rating_count, avg_rating, weighted_rating
    """
    logger.info(f"Getting top {n} popular movies (min_ratings={min_ratings})...")

    engine = get_sqlalchemy_engine()

    try:
        start_time = time.time()

        with engine.connect() as conn:
            # Use string formatting for parameters that cause issues with ::CAST syntax
            query_str = f"""
                WITH popular_movies AS (
                    SELECT
                        m.movie_id,
                        m.title,
                        m.release_year,
                        COUNT(r.rating_id) as rating_count,
                        AVG(r.rating) as avg_rating,
                        -- Weighted rating calculation (Bayesian Average)
                        -- WR = (v / (v + m)) * R + (m / (v + m)) * C
                        (COUNT(r.rating_id)::FLOAT / (COUNT(r.rating_id) + {min_ratings})) * AVG(r.rating) +
                        ({min_ratings}::FLOAT / (COUNT(r.rating_id) + {min_ratings})) * (SELECT AVG(rating) FROM ratings_train)
                        as weighted_rating
                    FROM movies m
                    INNER JOIN ratings_train r ON m.movie_id = r.movie_id
                    GROUP BY m.movie_id, m.title, m.release_year
                    HAVING COUNT(r.rating_id) >= {min_ratings}
                )
                SELECT
                    movie_id,
                    title,
                    release_year,
                    rating_count,
                    ROUND(avg_rating::NUMERIC, 2) as avg_rating,
                    ROUND(weighted_rating::NUMERIC, 2) as weighted_rating
                FROM popular_movies
                ORDER BY weighted_rating DESC, rating_count DESC
                LIMIT {n}
            """

            result_df = pd.read_sql(query_str, conn)

        elapsed_time = time.time() - start_time
        logger.info(f"Query executed in {elapsed_time:.3f} seconds")
        logger.info(f"Retrieved {len(result_df)} popular movies")

        return result_df

    except Exception as e:
        logger.error(f"Error getting popular movies: {e}")
        raise
    finally:
        engine.dispose()


def get_popular_movies_for_user(user_id: int, n: int = 10, min_ratings: int = 30) -> pd.DataFrame:
    """
    사용자 맞춤 인기 영화 추천 (이미 본 영화 제외)

    Args:
        user_id: 사용자 ID
        n: 추천할 영화 개수
        min_ratings: 최소 평점 개수

    Returns:
        pd.DataFrame: movie_id, title, release_year, rating_count, avg_rating, weighted_rating
    """
    logger.info(f"Getting top {n} popular movies for user {user_id} (min_ratings={min_ratings})...")

    engine = get_sqlalchemy_engine()

    try:
        start_time = time.time()

        with engine.connect() as conn:
            # Use string formatting for parameters that cause issues with ::CAST syntax
            query_str = f"""
                WITH user_watched AS (
                    SELECT movie_id
                    FROM ratings_train
                    WHERE user_id = {user_id}
                ),
                popular_movies AS (
                    SELECT
                        m.movie_id,
                        m.title,
                        m.release_year,
                        COUNT(r.rating_id) as rating_count,
                        AVG(r.rating) as avg_rating,
                        (COUNT(r.rating_id)::FLOAT / (COUNT(r.rating_id) + {min_ratings})) * AVG(r.rating) +
                        ({min_ratings}::FLOAT / (COUNT(r.rating_id) + {min_ratings})) * (SELECT AVG(rating) FROM ratings_train)
                        as weighted_rating
                    FROM movies m
                    INNER JOIN ratings_train r ON m.movie_id = r.movie_id
                    WHERE m.movie_id NOT IN (SELECT movie_id FROM user_watched)
                    GROUP BY m.movie_id, m.title, m.release_year
                    HAVING COUNT(r.rating_id) >= {min_ratings}
                )
                SELECT
                    movie_id,
                    title,
                    release_year,
                    rating_count,
                    ROUND(avg_rating::NUMERIC, 2) as avg_rating,
                    ROUND(weighted_rating::NUMERIC, 2) as weighted_rating
                FROM popular_movies
                ORDER BY weighted_rating DESC, rating_count DESC
                LIMIT {n}
            """

            result_df = pd.read_sql(query_str, conn)

        elapsed_time = time.time() - start_time
        logger.info(f"Query executed in {elapsed_time:.3f} seconds")
        logger.info(f"Retrieved {len(result_df)} popular movies for user {user_id}")

        return result_df

    except Exception as e:
        logger.error(f"Error getting popular movies for user {user_id}: {e}")
        raise
    finally:
        engine.dispose()


def get_recommendations_for_evaluation(user_ids: list, n: int = 10, min_ratings: int = 30) -> dict:
    """
    평가용 추천 결과 생성

    Args:
        user_ids: 사용자 ID 리스트
        n: 추천할 영화 개수
        min_ratings: 최소 평점 개수

    Returns:
        dict: {user_id: [movie_id 리스트]}
    """
    logger.info(f"Generating recommendations for {len(user_ids)} users...")

    recommendations = {}

    for i, user_id in enumerate(user_ids):
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i + 1}/{len(user_ids)} users")

        try:
            result_df = get_popular_movies_for_user(user_id, n=n, min_ratings=min_ratings)
            recommendations[user_id] = result_df['movie_id'].tolist()
        except Exception as e:
            logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
            recommendations[user_id] = []

    logger.info(f"Generated recommendations for {len(recommendations)} users")
    return recommendations


if __name__ == "__main__":
    # 테스트: 전체 인기 영화
    logger.info("=" * 60)
    logger.info("Test 1: Global Popular Movies")
    logger.info("=" * 60)

    popular_df = get_popular_movies(n=10, min_ratings=30)
    print("\nTop 10 Popular Movies:")
    print(popular_df.to_string(index=False))

    # 테스트: 사용자 맞춤 인기 영화
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: User-Personalized Popular Movies")
    logger.info("=" * 60)

    test_user_id = 1
    user_popular_df = get_popular_movies_for_user(user_id=test_user_id, n=10, min_ratings=30)
    print(f"\nTop 10 Popular Movies for User {test_user_id}:")
    print(user_popular_df.to_string(index=False))

    # 테스트: 평가용 추천 (샘플 10명)
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Batch Recommendations for Evaluation")
    logger.info("=" * 60)

    sample_users = list(range(1, 11))  # User 1-10
    batch_recommendations = get_recommendations_for_evaluation(sample_users, n=10, min_ratings=30)

    print(f"\nGenerated recommendations for {len(batch_recommendations)} users")
    print(f"Sample (User 1): {batch_recommendations[1][:5]}...")

    logger.info("\n" + "=" * 60)
    logger.info("[OK] All tests completed successfully!")
    logger.info("=" * 60)

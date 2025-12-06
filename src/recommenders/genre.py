"""
Genre-based Recommendation
장르 기반 추천 알고리즘
"""
import pandas as pd
import logging
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db_connection import get_sqlalchemy_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_user_genre_preference(user_id: int, min_ratings_per_genre: int = 3, top_n: int = 3) -> pd.DataFrame:
    """
    사용자 선호 장르 분석

    Args:
        user_id: 사용자 ID
        min_ratings_per_genre: 장르당 최소 평점 개수
        top_n: 상위 N개 장르 반환

    Returns:
        pd.DataFrame: genre_name, rating_count, avg_rating, preference_score
    """
    logger.info(f"Analyzing genre preference for user {user_id}...")

    engine = get_sqlalchemy_engine()

    try:
        start_time = time.time()

        with engine.connect() as conn:
            query_str = f"""
                SELECT
                    g.genre_name,
                    COUNT(r.rating_id) as rating_count,
                    ROUND(AVG(r.rating)::NUMERIC, 2) as avg_rating,
                    ROUND((COUNT(r.rating_id) * AVG(r.rating))::NUMERIC, 2) as preference_score
                FROM ratings_train r
                INNER JOIN movie_genres mg ON r.movie_id = mg.movie_id
                INNER JOIN genres g ON mg.genre_id = g.genre_id
                WHERE r.user_id = {user_id}
                GROUP BY g.genre_name
                HAVING COUNT(r.rating_id) >= {min_ratings_per_genre}
                ORDER BY preference_score DESC
                LIMIT {top_n}
            """

            result_df = pd.read_sql(query_str, conn)

        elapsed_time = time.time() - start_time
        logger.info(f"Query executed in {elapsed_time:.3f} seconds")
        logger.info(f"Found {len(result_df)} preferred genres for user {user_id}")

        return result_df

    except Exception as e:
        logger.error(f"Error analyzing genre preference for user {user_id}: {e}")
        raise
    finally:
        engine.dispose()


def get_genre_based_recommendations(
    user_id: int,
    n: int = 10,
    min_ratings_per_genre: int = 3,
    top_genres: int = 3,
    min_movie_ratings: int = 20
) -> pd.DataFrame:
    """
    장르 기반 영화 추천

    사용자가 선호하는 장르의 인기 영화를 추천

    Args:
        user_id: 사용자 ID
        n: 추천할 영화 개수
        min_ratings_per_genre: 장르당 최소 평점 개수
        top_genres: 사용할 상위 장르 개수
        min_movie_ratings: 영화당 최소 평점 개수

    Returns:
        pd.DataFrame: movie_id, title, release_year, preferred_genre, rating_count, avg_rating, weighted_rating, recommendation_score
    """
    logger.info(f"Getting genre-based recommendations for user {user_id} (n={n})...")

    engine = get_sqlalchemy_engine()

    try:
        start_time = time.time()

        with engine.connect() as conn:
            query_str = f"""
                WITH user_genre_preference AS (
                    SELECT
                        r.user_id,
                        g.genre_id,
                        g.genre_name,
                        COUNT(r.rating_id) as rating_count,
                        AVG(r.rating) as avg_rating,
                        COUNT(r.rating_id) * AVG(r.rating) as preference_score
                    FROM ratings_train r
                    INNER JOIN movie_genres mg ON r.movie_id = mg.movie_id
                    INNER JOIN genres g ON mg.genre_id = g.genre_id
                    WHERE r.user_id = {user_id}
                    GROUP BY r.user_id, g.genre_id, g.genre_name
                    HAVING COUNT(r.rating_id) >= {min_ratings_per_genre}
                    ORDER BY preference_score DESC
                    LIMIT {top_genres}
                ),
                user_watched AS (
                    SELECT movie_id
                    FROM ratings_train
                    WHERE user_id = {user_id}
                ),
                genre_movies AS (
                    SELECT DISTINCT
                        m.movie_id,
                        m.title,
                        m.release_year,
                        ugp.genre_name,
                        ugp.preference_score as genre_preference
                    FROM movies m
                    INNER JOIN movie_genres mg ON m.movie_id = mg.movie_id
                    INNER JOIN user_genre_preference ugp ON mg.genre_id = ugp.genre_id
                    WHERE m.movie_id NOT IN (SELECT movie_id FROM user_watched)
                ),
                movie_stats AS (
                    SELECT
                        gm.movie_id,
                        gm.title,
                        gm.release_year,
                        gm.genre_name,
                        gm.genre_preference,
                        COUNT(r.rating_id) as rating_count,
                        AVG(r.rating) as avg_rating,
                        (COUNT(r.rating_id)::FLOAT / (COUNT(r.rating_id) + {min_movie_ratings})) * AVG(r.rating) +
                        ({min_movie_ratings}::FLOAT / (COUNT(r.rating_id) + {min_movie_ratings})) * (SELECT AVG(rating) FROM ratings_train)
                        as weighted_rating,
                        ((COUNT(r.rating_id)::FLOAT / (COUNT(r.rating_id) + {min_movie_ratings})) * AVG(r.rating) +
                         ({min_movie_ratings}::FLOAT / (COUNT(r.rating_id) + {min_movie_ratings})) * (SELECT AVG(rating) FROM ratings_train))
                        * (gm.genre_preference / 100.0) as combined_score
                    FROM genre_movies gm
                    INNER JOIN ratings_train r ON gm.movie_id = r.movie_id
                    GROUP BY gm.movie_id, gm.title, gm.release_year, gm.genre_name, gm.genre_preference
                    HAVING COUNT(r.rating_id) >= {min_movie_ratings}
                )
                SELECT
                    ms.movie_id,
                    ms.title,
                    ms.release_year,
                    ms.genre_name as preferred_genre,
                    STRING_AGG(DISTINCT g2.genre_name, '|' ORDER BY g2.genre_name) as genres,
                    ms.rating_count,
                    ROUND(ms.avg_rating::NUMERIC, 2) as avg_rating,
                    ROUND(ms.weighted_rating::NUMERIC, 2) as weighted_rating,
                    ROUND(ms.combined_score::NUMERIC, 2) as combined_score
                FROM movie_stats ms
                LEFT JOIN movie_genres mg2 ON ms.movie_id = mg2.movie_id
                LEFT JOIN genres g2 ON mg2.genre_id = g2.genre_id
                GROUP BY ms.movie_id, ms.title, ms.release_year, ms.genre_name, ms.rating_count, ms.avg_rating, ms.weighted_rating, ms.combined_score
                ORDER BY ms.combined_score DESC, ms.weighted_rating DESC
                LIMIT {n}
            """

            result_df = pd.read_sql(query_str, conn)

        elapsed_time = time.time() - start_time
        logger.info(f"Query executed in {elapsed_time:.3f} seconds")
        logger.info(f"Retrieved {len(result_df)} genre-based recommendations for user {user_id}")

        return result_df

    except Exception as e:
        logger.error(f"Error getting genre-based recommendations for user {user_id}: {e}")
        raise
    finally:
        engine.dispose()


def get_recommendations_for_evaluation(user_ids: list, n: int = 10) -> dict:
    """
    평가용 장르 기반 추천 결과 생성

    Args:
        user_ids: 사용자 ID 리스트
        n: 추천할 영화 개수

    Returns:
        dict: {user_id: [movie_id 리스트]}
    """
    logger.info(f"Generating genre-based recommendations for {len(user_ids)} users...")

    recommendations = {}

    for i, user_id in enumerate(user_ids):
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i + 1}/{len(user_ids)} users")

        try:
            result_df = get_genre_based_recommendations(user_id, n=n)
            recommendations[user_id] = result_df['movie_id'].tolist()
        except Exception as e:
            logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
            recommendations[user_id] = []

    logger.info(f"Generated recommendations for {len(recommendations)} users")
    return recommendations


if __name__ == "__main__":
    # 테스트: 사용자 선호 장르 분석
    logger.info("=" * 60)
    logger.info("Test 1: User Genre Preference Analysis")
    logger.info("=" * 60)

    test_user_id = 1
    genre_pref_df = get_user_genre_preference(test_user_id, min_ratings_per_genre=3, top_n=5)
    print(f"\nTop 5 Preferred Genres for User {test_user_id}:")
    print(genre_pref_df.to_string(index=False))

    # 테스트: 장르 기반 추천
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Genre-based Recommendations")
    logger.info("=" * 60)

    genre_rec_df = get_genre_based_recommendations(test_user_id, n=10)
    print(f"\nTop 10 Genre-based Recommendations for User {test_user_id}:")
    print(genre_rec_df.to_string(index=False))

    # 테스트: 다른 사용자와 비교
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Compare Different Users")
    logger.info("=" * 60)

    test_user_2 = 100
    genre_pref_df_2 = get_user_genre_preference(test_user_2, min_ratings_per_genre=3, top_n=5)
    print(f"\nTop 5 Preferred Genres for User {test_user_2}:")
    print(genre_pref_df_2.to_string(index=False))

    genre_rec_df_2 = get_genre_based_recommendations(test_user_2, n=10)
    print(f"\nTop 10 Genre-based Recommendations for User {test_user_2}:")
    print(genre_rec_df_2.to_string(index=False))

    # 테스트: 평가용 배치 추천
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Batch Recommendations for Evaluation")
    logger.info("=" * 60)

    sample_users = list(range(1, 11))
    batch_recommendations = get_recommendations_for_evaluation(sample_users, n=10)

    print(f"\nGenerated recommendations for {len(batch_recommendations)} users")
    print(f"Sample (User 1): {batch_recommendations[1][:5]}...")
    print(f"Sample (User 10): {batch_recommendations[10][:5]}...")

    logger.info("\n" + "=" * 60)
    logger.info("[OK] All tests completed successfully!")
    logger.info("=" * 60)

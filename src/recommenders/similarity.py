"""
Item-based Collaborative Filtering - Similarity Calculation
아이템 기반 협업 필터링 - 유사도 계산
"""
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
import sys
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db_connection import get_sqlalchemy_engine
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_popular_movies(min_ratings: int = 100, limit: int = 500) -> List[int]:
    """
    Get list of popular movie IDs for similarity calculation
    인기 영화 ID 리스트 조회

    Args:
        min_ratings: Minimum number of ratings required
        limit: Maximum number of movies to return

    Returns:
        List[int]: Movie IDs sorted by rating count (descending)
    """
    logger.info(f"Fetching popular movies (min_ratings={min_ratings}, limit={limit})...")

    engine = get_sqlalchemy_engine()

    try:
        with engine.connect() as conn:
            # FIX: Use parameter binding to prevent SQL injection
            query = text("""
                SELECT
                    movie_id,
                    COUNT(*) as rating_count
                FROM ratings_train
                GROUP BY movie_id
                HAVING COUNT(*) >= :min_ratings
                ORDER BY COUNT(*) DESC
                LIMIT :limit
            """)

            result_df = pd.read_sql(query, conn, params={'min_ratings': min_ratings, 'limit': limit})

        movie_ids = result_df['movie_id'].tolist()
        logger.info(f"Found {len(movie_ids)} popular movies")

        return movie_ids

    except Exception as e:
        logger.error(f"Error fetching popular movies: {e}")
        raise
    finally:
        engine.dispose()


def calculate_similarity_for_pair(
    movie_id_1: int,
    movie_id_2: int,
    min_common_users: int = 20
) -> Tuple[int, int, int, float]:
    """
    Calculate cosine similarity between two movies
    두 영화 간 코사인 유사도 계산

    DEPRECATED: Use calculate_similarities_optimized() instead for better performance
    이 함수는 성능상의 이유로 사용하지 않는 것을 권장합니다

    Args:
        movie_id_1: First movie ID
        movie_id_2: Second movie ID
        min_common_users: Minimum number of common users required

    Returns:
        Tuple[int, int, int, float]: (movie_id_1, movie_id_2, common_users, similarity)
        Returns None if insufficient common users
    """
    engine = get_sqlalchemy_engine()

    try:
        with engine.connect() as conn:
            # FIX: Use parameter binding to prevent SQL injection
            query = text("""
                WITH common_ratings AS (
                    SELECT
                        r1.user_id,
                        r1.rating as rating_1,
                        r2.rating as rating_2
                    FROM ratings_train r1
                    INNER JOIN ratings_train r2
                        ON r1.user_id = r2.user_id
                    WHERE r1.movie_id = :movie_id_1
                      AND r2.movie_id = :movie_id_2
                ),
                similarity_calc AS (
                    SELECT
                        COUNT(*) as common_users,
                        SUM(rating_1 * rating_2) as dot_product,
                        SQRT(SUM(rating_1 * rating_1)) as magnitude_1,
                        SQRT(SUM(rating_2 * rating_2)) as magnitude_2
                    FROM common_ratings
                )
                SELECT
                    common_users,
                    CASE
                        WHEN magnitude_1 > 0 AND magnitude_2 > 0 THEN
                            dot_product / (magnitude_1 * magnitude_2)
                        ELSE 0
                    END as similarity_score
                FROM similarity_calc
                WHERE common_users >= :min_common_users
            """)

            result_df = pd.read_sql(query, conn, params={
                'movie_id_1': movie_id_1,
                'movie_id_2': movie_id_2,
                'min_common_users': min_common_users
            })

        if len(result_df) == 0:
            return None

        row = result_df.iloc[0]
        return (movie_id_1, movie_id_2, int(row['common_users']), float(row['similarity_score']))

    except Exception as e:
        logger.error(f"Error calculating similarity for ({movie_id_1}, {movie_id_2}): {e}")
        return None
    finally:
        engine.dispose()


def calculate_similarities_batch(
    movie_ids: List[int],
    min_common_users: int = 20,
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Calculate similarities for all pairs of movies in batches
    영화 쌍의 유사도를 배치로 계산

    Args:
        movie_ids: List of movie IDs to calculate similarities for
        min_common_users: Minimum number of common users required
        batch_size: Number of pairs to process before progress update

    Returns:
        pd.DataFrame: Columns [movie_id_1, movie_id_2, common_users, similarity_score]
    """
    logger.info(f"Calculating similarities for {len(movie_ids)} movies...")

    # Generate all pairs (movie_id_1 < movie_id_2 to avoid duplicates)
    pairs = []
    for i in range(len(movie_ids)):
        for j in range(i + 1, len(movie_ids)):
            pairs.append((movie_ids[i], movie_ids[j]))

    total_pairs = len(pairs)
    logger.info(f"Total pairs to calculate: {total_pairs:,}")

    similarities = []
    start_time = time.time()

    for idx, (movie_1, movie_2) in enumerate(pairs):
        # Calculate similarity
        result = calculate_similarity_for_pair(movie_1, movie_2, min_common_users)

        if result is not None:
            similarities.append(result)

        # Progress update
        if (idx + 1) % batch_size == 0:
            elapsed = time.time() - start_time
            pairs_per_sec = (idx + 1) / elapsed
            remaining_pairs = total_pairs - (idx + 1)
            eta_seconds = remaining_pairs / pairs_per_sec if pairs_per_sec > 0 else 0

            logger.info(
                f"Progress: {idx + 1:,}/{total_pairs:,} pairs "
                f"({(idx + 1) / total_pairs * 100:.1f}%) | "
                f"Found {len(similarities):,} similarities | "
                f"Speed: {pairs_per_sec:.1f} pairs/sec | "
                f"ETA: {eta_seconds / 60:.1f} min"
            )

    elapsed_time = time.time() - start_time
    logger.info(f"Calculation completed in {elapsed_time / 60:.1f} minutes")
    logger.info(f"Found {len(similarities):,} valid similarities out of {total_pairs:,} pairs")

    # Convert to DataFrame
    similarity_df = pd.DataFrame(
        similarities,
        columns=['movie_id_1', 'movie_id_2', 'common_users', 'similarity_score']
    )

    return similarity_df


def calculate_similarities_optimized(
    movie_ids: List[int],
    min_common_users: int = 20
) -> pd.DataFrame:
    """
    Optimized batch similarity calculation using single SQL query
    단일 SQL 쿼리로 최적화된 배치 유사도 계산

    This is much faster than pair-by-pair calculation

    Args:
        movie_ids: List of movie IDs
        min_common_users: Minimum number of common users

    Returns:
        pd.DataFrame: Similarity scores with columns [movie_id_1, movie_id_2, common_users, similarity_score]
    """
    logger.info(f"Calculating similarities for {len(movie_ids)} movies (optimized)...")

    engine = get_sqlalchemy_engine()

    try:
        start_time = time.time()

        with engine.connect() as conn:
            # FIX: Use ANY array instead of string concatenation to prevent SQL injection
            query = text("""
                WITH movie_pairs AS (
                    -- Generate all pairs (m1 < m2 to avoid duplicates)
                    SELECT
                        m1.movie_id as movie_id_1,
                        m2.movie_id as movie_id_2
                    FROM (SELECT DISTINCT movie_id FROM ratings_train WHERE movie_id = ANY(:movie_ids)) m1
                    CROSS JOIN (SELECT DISTINCT movie_id FROM ratings_train WHERE movie_id = ANY(:movie_ids)) m2
                    WHERE m1.movie_id < m2.movie_id
                ),
                pair_similarities AS (
                    SELECT
                        mp.movie_id_1,
                        mp.movie_id_2,
                        COUNT(r1.user_id) as common_users,
                        SUM(r1.rating * r2.rating) as dot_product,
                        SQRT(SUM(r1.rating * r1.rating)) as magnitude_1,
                        SQRT(SUM(r2.rating * r2.rating)) as magnitude_2
                    FROM movie_pairs mp
                    INNER JOIN ratings_train r1 ON r1.movie_id = mp.movie_id_1
                    INNER JOIN ratings_train r2
                        ON r2.movie_id = mp.movie_id_2
                        AND r2.user_id = r1.user_id
                    GROUP BY mp.movie_id_1, mp.movie_id_2
                    HAVING COUNT(r1.user_id) >= :min_common_users
                )
                SELECT
                    movie_id_1,
                    movie_id_2,
                    common_users,
                    CASE
                        WHEN magnitude_1 > 0 AND magnitude_2 > 0 THEN
                            ROUND((dot_product / (magnitude_1 * magnitude_2))::NUMERIC, 4)
                        ELSE 0
                    END as similarity_score
                FROM pair_similarities
                WHERE magnitude_1 > 0 AND magnitude_2 > 0
                ORDER BY similarity_score DESC
            """)

            result_df = pd.read_sql(query, conn, params={
                'movie_ids': movie_ids,
                'min_common_users': min_common_users
            })

        elapsed_time = time.time() - start_time
        logger.info(f"Query completed in {elapsed_time:.1f} seconds ({elapsed_time / 60:.1f} minutes)")
        logger.info(f"Found {len(result_df):,} valid similarities")

        return result_df

    except Exception as e:
        logger.error(f"Error in optimized similarity calculation: {e}")
        raise
    finally:
        engine.dispose()


def save_similarities_to_db(similarity_df: pd.DataFrame, table_name: str = 'movie_similarities'):
    """
    Save similarity scores to database
    유사도 점수를 데이터베이스에 저장

    Args:
        similarity_df: DataFrame with columns [movie_id_1, movie_id_2, common_users, similarity_score]
        table_name: Target table name
    """
    logger.info(f"Saving {len(similarity_df):,} similarities to {table_name}...")

    engine = get_sqlalchemy_engine()

    try:
        start_time = time.time()

        # Create table if not exists
        with engine.connect() as conn:
            create_table_query = text(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    similarity_id SERIAL PRIMARY KEY,
                    movie_id_1 INTEGER NOT NULL,
                    movie_id_2 INTEGER NOT NULL,
                    common_users INTEGER NOT NULL,
                    similarity_score FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(movie_id_1, movie_id_2)
                )
            """)
            conn.execute(create_table_query)
            conn.commit()

            # Create indexes for faster lookups
            index_queries = [
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_movie1 ON {table_name}(movie_id_1)",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_movie2 ON {table_name}(movie_id_2)",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_score ON {table_name}(similarity_score DESC)"
            ]

            for idx_query in index_queries:
                conn.execute(text(idx_query))

            conn.commit()

        # Save data using pandas to_sql
        similarity_df.to_sql(
            table_name,
            engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Saved to database in {elapsed_time:.1f} seconds")

    except Exception as e:
        logger.error(f"Error saving similarities to database: {e}")
        raise
    finally:
        engine.dispose()


def calculate_and_save_all_similarities(
    min_ratings: int = 100,
    limit: int = 500,
    min_common_users: int = 20,
    use_optimized: bool = True
):
    """
    Main function: Calculate all similarities and save to database
    메인 함수: 모든 유사도 계산 후 데이터베이스 저장

    Args:
        min_ratings: Minimum ratings for a movie to be included
        limit: Maximum number of movies to process
        min_common_users: Minimum common users for similarity calculation
        use_optimized: Use optimized single-query method (recommended)
    """
    logger.info("=" * 60)
    logger.info("ITEM-BASED CF: SIMILARITY CALCULATION")
    logger.info("=" * 60)

    # Step 1: Get popular movies
    movie_ids = get_popular_movies(min_ratings=min_ratings, limit=limit)

    total_pairs = len(movie_ids) * (len(movie_ids) - 1) // 2
    logger.info(f"\nProcessing {len(movie_ids)} movies → {total_pairs:,} pairs")

    # Step 2: Calculate similarities
    if use_optimized:
        logger.info("\nUsing optimized batch calculation...")
        similarity_df = calculate_similarities_optimized(movie_ids, min_common_users)
    else:
        logger.info("\nUsing pair-by-pair calculation (slower)...")
        similarity_df = calculate_similarities_batch(movie_ids, min_common_users)

    # Step 3: Save to database
    logger.info("\n" + "=" * 60)
    save_similarities_to_db(similarity_df)

    # Step 4: Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Movies processed: {len(movie_ids)}")
    logger.info(f"Total possible pairs: {total_pairs:,}")
    logger.info(f"Valid similarities found: {len(similarity_df):,}")
    logger.info(f"Coverage: {len(similarity_df) / total_pairs * 100:.1f}%")
    logger.info(f"\nSimilarity score statistics:")
    logger.info(f"  Mean: {similarity_df['similarity_score'].mean():.4f}")
    logger.info(f"  Median: {similarity_df['similarity_score'].median():.4f}")
    logger.info(f"  Min: {similarity_df['similarity_score'].min():.4f}")
    logger.info(f"  Max: {similarity_df['similarity_score'].max():.4f}")
    logger.info(f"\nCommon users statistics:")
    logger.info(f"  Mean: {similarity_df['common_users'].mean():.1f}")
    logger.info(f"  Median: {similarity_df['common_users'].median():.1f}")
    logger.info(f"  Min: {similarity_df['common_users'].min()}")
    logger.info(f"  Max: {similarity_df['common_users'].max()}")
    logger.info("=" * 60)

    return similarity_df


def get_similar_movies_for_movie(
    movie_id: int,
    n: int = 10,
    table_name: str = 'movie_similarities'
) -> pd.DataFrame:
    """
    Get N most similar movies for a given movie
    특정 영화와 유사한 영화 추천

    Args:
        movie_id: Target movie ID
        n: Number of recommendations
        table_name: Similarity table name

    Returns:
        pd.DataFrame: Similar movies with columns [movie_id, title, genres, similarity_score, common_users]
    """
    logger.info(f"Finding similar movies for movie_id={movie_id}, top_n={n}...")

    engine = get_sqlalchemy_engine()

    try:
        start_time = time.time()

        with engine.connect() as conn:
            query = text("""
                SELECT
                    CASE
                        WHEN ms.movie_id_1 = :movie_id THEN ms.movie_id_2
                        ELSE ms.movie_id_1
                    END as movie_id,
                    m.title,
                    STRING_AGG(DISTINCT g.genre_name, '|' ORDER BY g.genre_name) as genres,
                    ms.similarity_score
                FROM movie_similarities ms
                INNER JOIN movies m ON m.movie_id = CASE
                    WHEN ms.movie_id_1 = :movie_id THEN ms.movie_id_2
                    ELSE ms.movie_id_1
                END
                LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.genre_id
                WHERE ms.movie_id_1 = :movie_id OR ms.movie_id_2 = :movie_id
                GROUP BY
                    CASE WHEN ms.movie_id_1 = :movie_id THEN ms.movie_id_2 ELSE ms.movie_id_1 END,
                    m.title,
                    ms.similarity_score
                ORDER BY ms.similarity_score DESC
                LIMIT :n
            """)

            result_df = pd.read_sql(query, conn, params={'movie_id': movie_id, 'n': n})

        elapsed_time = time.time() - start_time
        logger.info(f"Query executed in {elapsed_time:.3f} seconds")
        logger.info(f"Found {len(result_df)} similar movies")

        return result_df

    except Exception as e:
        logger.error(f"Error fetching similar movies: {e}")
        raise
    finally:
        engine.dispose()


def get_similar_movies_for_user(
    user_id: int,
    n: int = 10,
    min_rating: float = 4.0,
    table_name: str = 'movie_similarities'
) -> pd.DataFrame:
    """
    Get personalized movie recommendations based on item-based collaborative filtering
    사용자 기반 유사 영화 추천 (Item-based CF)

    Algorithm:
    1. Get movies user rated highly (>= min_rating)
    2. Find similar movies to those liked movies
    3. Aggregate similarity scores (weighted by user's rating)
    4. Exclude already watched movies
    5. Return top N recommendations

    Args:
        user_id: Target user ID
        n: Number of recommendations
        min_rating: Minimum rating threshold for "liked" movies
        table_name: Similarity table name

    Returns:
        pd.DataFrame: Recommended movies with columns [movie_id, title, genres, recommendation_score, based_on_movies]
    """
    logger.info(f"Generating recommendations for user_id={user_id}, top_n={n}...")

    engine = get_sqlalchemy_engine()

    try:
        start_time = time.time()

        with engine.connect() as conn:
            query = text("""
                WITH user_liked_movies AS (
                    -- Movies user rated highly
                    SELECT
                        movie_id,
                        rating
                    FROM ratings_train
                    WHERE user_id = :user_id
                      AND rating >= :min_rating
                ),
                user_watched AS (
                    -- All movies user has watched (to exclude from recommendations)
                    SELECT movie_id
                    FROM ratings_train
                    WHERE user_id = :user_id
                ),
                similar_candidates AS (
                    -- Find movies similar to liked movies
                    SELECT
                        CASE
                            WHEN ms.movie_id_1 IN (SELECT movie_id FROM user_liked_movies) THEN ms.movie_id_2
                            ELSE ms.movie_id_1
                        END as recommended_movie_id,
                        CASE
                            WHEN ms.movie_id_1 IN (SELECT movie_id FROM user_liked_movies) THEN ms.movie_id_1
                            ELSE ms.movie_id_2
                        END as source_movie_id,
                        ms.similarity_score,
                        ulm.rating as user_rating
                    FROM movie_similarities ms
                    INNER JOIN user_liked_movies ulm
                        ON (ms.movie_id_1 = ulm.movie_id OR ms.movie_id_2 = ulm.movie_id)
                    WHERE
                        (ms.movie_id_1 IN (SELECT movie_id FROM user_liked_movies)
                         OR ms.movie_id_2 IN (SELECT movie_id FROM user_liked_movies))
                        -- Exclude already watched movies
                        AND CASE
                            WHEN ms.movie_id_1 IN (SELECT movie_id FROM user_liked_movies) THEN ms.movie_id_2
                            ELSE ms.movie_id_1
                        END NOT IN (SELECT movie_id FROM user_watched)
                ),
                aggregated_scores AS (
                    -- Aggregate scores: weight similarity by user's rating
                    SELECT
                        recommended_movie_id,
                        SUM(similarity_score * (user_rating / 5.0)) as recommendation_score,
                        COUNT(DISTINCT source_movie_id) as based_on_count,
                        STRING_AGG(DISTINCT source_movie_id::TEXT, ',' ORDER BY source_movie_id::TEXT) as based_on_movies
                    FROM similar_candidates
                    GROUP BY recommended_movie_id
                )
                SELECT
                    agg.recommended_movie_id as movie_id,
                    m.title,
                    STRING_AGG(DISTINCT g.genre_name, '|' ORDER BY g.genre_name) as genres,
                    ROUND(agg.recommendation_score::NUMERIC, 4) as recommendation_score,
                    agg.based_on_count,
                    agg.based_on_movies
                FROM aggregated_scores agg
                INNER JOIN movies m ON m.movie_id = agg.recommended_movie_id
                LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.genre_id
                GROUP BY
                    agg.recommended_movie_id,
                    m.title,
                    agg.recommendation_score,
                    agg.based_on_count,
                    agg.based_on_movies
                ORDER BY agg.recommendation_score DESC
                LIMIT :n
            """)

            result_df = pd.read_sql(query, conn, params={
                'user_id': user_id,
                'min_rating': min_rating,
                'n': n
            })

        elapsed_time = time.time() - start_time
        logger.info(f"Query executed in {elapsed_time:.3f} seconds")
        logger.info(f"Found {len(result_df)} recommendations")

        return result_df

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise
    finally:
        engine.dispose()


def get_recommendations_for_evaluation(
    user_ids: List[int],
    n: int = 10,
    min_rating: float = 4.0
) -> dict:
    """
    Get recommendations for multiple users (for batch evaluation)
    배치 평가용 추천 결과 생성

    Args:
        user_ids: List of user IDs
        n: Number of recommendations per user
        min_rating: Minimum rating threshold for "liked" movies

    Returns:
        dict: {user_id: [movie_id_1, movie_id_2, ...]}
    """
    logger.info(f"Generating recommendations for {len(user_ids)} users (batch evaluation)...")

    recommendations = {}
    start_time = time.time()

    for idx, user_id in enumerate(user_ids):
        try:
            result_df = get_similar_movies_for_user(user_id, n=n, min_rating=min_rating)
            recommendations[user_id] = result_df['movie_id'].tolist()
        except Exception as e:
            logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
            recommendations[user_id] = []

        # Progress logging
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            users_per_sec = (idx + 1) / elapsed
            remaining_users = len(user_ids) - (idx + 1)
            eta_seconds = remaining_users / users_per_sec if users_per_sec > 0 else 0

            logger.info(
                f"Progress: {idx + 1}/{len(user_ids)} users "
                f"({(idx + 1) / len(user_ids) * 100:.1f}%) | "
                f"Speed: {users_per_sec:.1f} users/sec | "
                f"ETA: {eta_seconds / 60:.1f} min"
            )

    elapsed_time = time.time() - start_time
    logger.info(f"Batch evaluation completed in {elapsed_time / 60:.1f} minutes")
    logger.info(f"Generated recommendations for {len(recommendations)} users")

    return recommendations


if __name__ == "__main__":
    # Test 1: Small test with top 50 movies
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Small Scale Test (50 movies)")
    logger.info("=" * 60)

    test_similarity_df = calculate_and_save_all_similarities(
        min_ratings=100,
        limit=50,
        min_common_users=20,
        use_optimized=True
    )

    print("\n[Sample] Top 10 Most Similar Movie Pairs:")
    print(test_similarity_df.head(10).to_string(index=False))

    # Uncomment to run full calculation (500 movies - takes ~10-30 minutes)
    # logger.info("\n" + "=" * 60)
    # logger.info("FULL SCALE: Calculating similarities for 500 movies")
    # logger.info("=" * 60)
    #
    # full_similarity_df = calculate_and_save_all_similarities(
    #     min_ratings=100,
    #     limit=500,
    #     min_common_users=20,
    #     use_optimized=True
    # )

    logger.info("\n[OK] Similarity calculation completed!")

"""
Evaluate Item-based Collaborative Filtering Recommendations
Item-based CF 추천 알고리즘 평가
"""
import logging
import time
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recommenders.similarity import get_recommendations_for_evaluation
from src.evaluator import (
    calculate_hit_rate_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k
)
from src.db_connection import get_sqlalchemy_engine
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_test_users(limit: int = 1000, min_ratings: int = 20) -> list:
    """
    Get list of test users who have sufficient ratings
    충분한 평점을 가진 테스트 사용자 조회

    Args:
        limit: Maximum number of users
        min_ratings: Minimum number of ratings required

    Returns:
        list: User IDs
    """
    logger.info(f"Fetching {limit} test users with min_ratings={min_ratings}...")

    engine = get_sqlalchemy_engine()

    try:
        with engine.connect() as conn:
            query = text("""
                SELECT
                    user_id,
                    COUNT(*) as rating_count
                FROM ratings_test
                GROUP BY user_id
                HAVING COUNT(*) >= :min_ratings
                ORDER BY COUNT(*) DESC
                LIMIT :limit
            """)

            result_df = pd.read_sql(query, conn, params={'min_ratings': min_ratings, 'limit': limit})

        user_ids = result_df['user_id'].tolist()
        logger.info(f"Found {len(user_ids)} test users")

        return user_ids

    except Exception as e:
        logger.error(f"Error fetching test users: {e}")
        raise
    finally:
        engine.dispose()


def get_ground_truth(user_ids: list, min_rating: float = 4.0) -> dict:
    """
    Get ground truth (actual highly-rated movies in test set)
    실제 정답 데이터 조회 (테스트 셋에서 높은 평점을 준 영화)

    Args:
        user_ids: List of user IDs
        min_rating: Minimum rating to consider as "liked"

    Returns:
        dict: {user_id: [movie_id_1, movie_id_2, ...]}
    """
    logger.info(f"Fetching ground truth for {len(user_ids)} users...")

    engine = get_sqlalchemy_engine()

    try:
        with engine.connect() as conn:
            query = text("""
                SELECT
                    user_id,
                    movie_id
                FROM ratings_test
                WHERE user_id = ANY(:user_ids)
                  AND rating >= :min_rating
                ORDER BY user_id, rating DESC
            """)

            result_df = pd.read_sql(query, conn, params={
                'user_ids': user_ids,
                'min_rating': min_rating
            })

        # Convert to dict
        ground_truth = {}
        for user_id in user_ids:
            user_movies = result_df[result_df['user_id'] == user_id]['movie_id'].tolist()
            ground_truth[user_id] = user_movies

        logger.info(f"Ground truth collected for {len(ground_truth)} users")

        return ground_truth

    except Exception as e:
        logger.error(f"Error fetching ground truth: {e}")
        raise
    finally:
        engine.dispose()


def evaluate_similarity_recommendations(
    n_users: int = 1000,
    k: int = 10,
    min_rating: float = 4.0,
    min_test_ratings: int = 20
):
    """
    Evaluate Item-based CF recommendations
    Item-based CF 추천 평가

    Args:
        n_users: Number of users to evaluate
        k: Number of recommendations per user
        min_rating: Minimum rating threshold for liked movies
        min_test_ratings: Minimum test ratings required per user

    Returns:
        dict: Evaluation metrics
    """
    logger.info("=" * 60)
    logger.info("ITEM-BASED COLLABORATIVE FILTERING EVALUATION")
    logger.info("=" * 60)

    # Step 1: Get test users
    logger.info(f"\n[Step 1] Fetching {n_users} test users...")
    user_ids = get_test_users(limit=n_users, min_ratings=min_test_ratings)
    logger.info(f"Selected {len(user_ids)} users for evaluation")

    # Step 2: Get ground truth
    logger.info(f"\n[Step 2] Fetching ground truth...")
    ground_truth = get_ground_truth(user_ids, min_rating=min_rating)

    # Calculate ground truth statistics
    total_relevant = sum(len(movies) for movies in ground_truth.values())
    avg_relevant = total_relevant / len(user_ids) if user_ids else 0
    logger.info(f"Total relevant movies: {total_relevant}")
    logger.info(f"Average relevant movies per user: {avg_relevant:.2f}")

    # Step 3: Generate recommendations
    logger.info(f"\n[Step 3] Generating item-based CF recommendations...")
    start_time = time.time()
    recommendations = get_recommendations_for_evaluation(
        user_ids=user_ids,
        n=k,
        min_rating=min_rating
    )
    elapsed_time = time.time() - start_time

    # Step 4: Calculate metrics
    logger.info(f"\n[Step 4] Calculating evaluation metrics...")
    hit_rate = calculate_hit_rate_at_k(recommendations, ground_truth, k)
    precision = calculate_precision_at_k(recommendations, ground_truth, k)
    recall = calculate_recall_at_k(recommendations, ground_truth, k)

    # Average latency per user
    avg_latency = (elapsed_time / len(user_ids)) * 1000  # milliseconds

    # Step 5: Print results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Algorithm: Item-based Collaborative Filtering")
    logger.info(f"Users evaluated: {len(user_ids)}")
    logger.info(f"K (recommendations per user): {k}")
    logger.info(f"Min rating threshold: {min_rating}")
    logger.info(f"\nMetrics:")
    logger.info(f"  Hit Rate@{k}: {hit_rate:.4f} ({hit_rate * 100:.2f}%)")
    logger.info(f"  Precision@{k}: {precision:.4f} ({precision * 100:.2f}%)")
    logger.info(f"  Recall@{k}: {recall:.4f} ({recall * 100:.2f}%)")
    logger.info(f"\nPerformance:")
    logger.info(f"  Total time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    logger.info(f"  Average latency: {avg_latency:.0f}ms per user")
    logger.info(f"  Users per second: {len(user_ids) / elapsed_time:.2f}")
    logger.info("=" * 60)

    # Return results
    results = {
        'algorithm': 'Item-based CF',
        'n_users': len(user_ids),
        'k': k,
        'min_rating': min_rating,
        'hit_rate': hit_rate,
        'precision': precision,
        'recall': recall,
        'total_time': elapsed_time,
        'avg_latency_ms': avg_latency
    }

    return results


if __name__ == "__main__":
    # Evaluate with 1,000 users, K=10
    results = evaluate_similarity_recommendations(
        n_users=1000,
        k=10,
        min_rating=4.0,
        min_test_ratings=20
    )

    print("\n[OK] Evaluation completed!")
    print(f"Hit Rate@10: {results['hit_rate']:.4f}")
    print(f"Precision@10: {results['precision']:.4f}")
    print(f"Recall@10: {results['recall']:.4f}")

"""
Evaluate Hybrid Recommendation System
하이브리드 추천 시스템 평가
"""
import logging
import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommenders.hybrid import get_recommendations_for_evaluation
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
    """Get test users"""
    logger.info(f"Fetching {limit} test users...")

    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT user_id, COUNT(*) as rating_count
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
    """Get ground truth from test set"""
    logger.info(f"Fetching ground truth for {len(user_ids)} users...")

    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT user_id, movie_id
                FROM ratings_test
                WHERE user_id = ANY(:user_ids)
                  AND rating >= :min_rating
                ORDER BY user_id, rating DESC
            """)
            result_df = pd.read_sql(query, conn, params={
                'user_ids': user_ids,
                'min_rating': min_rating
            })

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


def evaluate_hybrid_recommendations(
    n_users: int = 1000,
    k: int = 10,
    min_rating: float = 4.0,
    min_test_ratings: int = 20,
    weights: dict = None,
    model_path: str = 'models/svd_model.pkl'
):
    """
    Evaluate hybrid recommendations
    하이브리드 추천 평가

    Args:
        n_users: Number of users to evaluate
        k: Number of recommendations per user
        min_rating: Minimum rating threshold for ground truth
        min_test_ratings: Minimum test ratings per user
        weights: Algorithm weights (default: {'popularity': 0.1, 'genre': 0.2, 'similarity': 0.3, 'ml': 0.4})
        model_path: Path to trained ML model
    """
    if weights is None:
        weights = {'popularity': 0.1, 'genre': 0.2, 'similarity': 0.3, 'ml': 0.4}

    logger.info("=" * 60)
    logger.info("HYBRID RECOMMENDATION EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Weights: {weights}")

    # Step 1: Get test users
    logger.info(f"\n[Step 1] Fetching {n_users} test users...")
    user_ids = get_test_users(limit=n_users, min_ratings=min_test_ratings)

    # Step 2: Get ground truth
    logger.info(f"\n[Step 2] Fetching ground truth...")
    ground_truth = get_ground_truth(user_ids, min_rating=min_rating)

    total_relevant = sum(len(movies) for movies in ground_truth.values())
    avg_relevant = total_relevant / len(user_ids) if user_ids else 0
    logger.info(f"Total relevant movies: {total_relevant}")
    logger.info(f"Average relevant movies per user: {avg_relevant:.2f}")

    # Step 3: Generate recommendations
    logger.info(f"\n[Step 3] Generating hybrid recommendations...")
    start_time = time.time()
    recommendations = get_recommendations_for_evaluation(
        user_ids=user_ids,
        n=k,
        weights=weights,
        ml_model_path=model_path
    )
    elapsed_time = time.time() - start_time

    # Step 4: Calculate metrics
    logger.info(f"\n[Step 4] Calculating evaluation metrics...")

    hit_rate = calculate_hit_rate_at_k(recommendations, ground_truth, k)
    precision = calculate_precision_at_k(recommendations, ground_truth, k)
    recall = calculate_recall_at_k(recommendations, ground_truth, k)
    avg_latency = (elapsed_time / len(user_ids)) * 1000

    # Step 5: Print results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Algorithm: Hybrid")
    logger.info(f"Weights: {weights}")
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

    return {
        'algorithm': 'Hybrid',
        'weights': weights,
        'n_users': len(user_ids),
        'k': k,
        'hit_rate': hit_rate,
        'precision': precision,
        'recall': recall,
        'total_time': elapsed_time,
        'avg_latency_ms': avg_latency
    }


def compare_weight_configurations(
    n_users: int = 500,
    k: int = 10,
    model_path: str = 'models/svd_model.pkl'
):
    """
    Compare different weight configurations
    다양한 가중치 설정 비교

    Args:
        n_users: Number of users to evaluate
        k: Number of recommendations per user
        model_path: Path to trained ML model
    """
    logger.info("\n" + "=" * 70)
    logger.info("COMPARING WEIGHT CONFIGURATIONS")
    logger.info("=" * 70)

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
        },
        {
            'name': 'Genre-focused',
            'weights': {'popularity': 0.1, 'genre': 0.5, 'similarity': 0.2, 'ml': 0.2}
        },
        {
            'name': 'Pure ML',
            'weights': {'popularity': 0.0, 'genre': 0.0, 'similarity': 0.0, 'ml': 1.0}
        }
    ]

    results = []

    for config in weight_configs:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Testing: {config['name']}")
        logger.info(f"Weights: {config['weights']}")
        logger.info("=" * 70)

        try:
            result = evaluate_hybrid_recommendations(
                n_users=n_users,
                k=k,
                weights=config['weights'],
                model_path=model_path
            )
            result['config_name'] = config['name']
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to evaluate {config['name']}: {e}", exc_info=True)

    # Print comparison table
    if results:
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON TABLE")
        logger.info("=" * 70)

        df = pd.DataFrame(results)
        df = df[['config_name', 'hit_rate', 'precision', 'recall', 'avg_latency_ms']]
        df['hit_rate'] = df['hit_rate'].apply(lambda x: f"{x:.4f}")
        df['precision'] = df['precision'].apply(lambda x: f"{x:.4f}")
        df['recall'] = df['recall'].apply(lambda x: f"{x:.4f}")
        df['avg_latency_ms'] = df['avg_latency_ms'].apply(lambda x: f"{x:.0f}")

        print("\n" + df.to_string(index=False))
        print("=" * 70)

        # Find best configuration
        results_numeric = pd.DataFrame(results)
        best_hit_rate = results_numeric.loc[results_numeric['hit_rate'].idxmax()]
        best_precision = results_numeric.loc[results_numeric['precision'].idxmax()]
        best_recall = results_numeric.loc[results_numeric['recall'].idxmax()]

        logger.info("\n[BEST CONFIGURATIONS]")
        logger.info(f"Best Hit Rate: {best_hit_rate['config_name']} ({best_hit_rate['hit_rate']:.4f})")
        logger.info(f"Best Precision: {best_precision['config_name']} ({best_precision['precision']:.4f})")
        logger.info(f"Best Recall: {best_recall['config_name']} ({best_recall['recall']:.4f})")

    return results


if __name__ == "__main__":
    # Check if model exists
    model_path = Path('models/svd_model.pkl')

    if not model_path.exists():
        logger.error("Model not found! Please run: python src/recommenders/ml_based.py")
        sys.exit(1)

    # Option 1: Evaluate with default weights
    logger.info("\n[Option 1] Evaluating with default weights...")
    results_default = evaluate_hybrid_recommendations(
        n_users=1000,
        k=10,
        min_rating=4.0,
        min_test_ratings=20,
        model_path=str(model_path)
    )

    print("\n[OK] Hybrid evaluation completed!")
    print(f"Hit Rate@10: {results_default['hit_rate']:.4f}")
    print(f"Precision@10: {results_default['precision']:.4f}")
    print(f"Recall@10: {results_default['recall']:.4f}")

    # Option 2: Compare weight configurations
    logger.info("\n" + "=" * 70)
    logger.info("[Option 2] Comparing weight configurations...")
    logger.info("=" * 70)

    compare_results = compare_weight_configurations(
        n_users=500,
        k=10,
        model_path=str(model_path)
    )

    print("\n[OK] Weight configuration comparison completed!")

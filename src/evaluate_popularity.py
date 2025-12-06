"""
Evaluate popularity-based recommendation algorithm
인기 기반 추천 알고리즘 평가
"""
import pandas as pd
import time
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recommenders.popularity import get_recommendations_for_evaluation
from src.evaluator import evaluate_recommendations
from src.db_connection import get_sqlalchemy_engine
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data():
    """
    Load test set data for evaluation

    Returns:
        dict: {user_id: [movie_id list]}
    """
    logger.info("Loading test data...")

    engine = get_sqlalchemy_engine()

    try:
        with engine.connect() as conn:
            query = text("""
                SELECT
                    user_id,
                    movie_id
                FROM ratings_test
                WHERE rating >= 4.0  -- Consider only positive ratings
                ORDER BY user_id, timestamp
            """)

            test_df = pd.read_sql(query, conn)

        # Convert to dict format
        test_items = test_df.groupby('user_id')['movie_id'].apply(list).to_dict()

        logger.info(f"Loaded test data for {len(test_items)} users")
        return test_items

    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise
    finally:
        engine.dispose()


def benchmark_performance(sample_size=100):
    """
    Benchmark query performance

    Args:
        sample_size: Number of users to test

    Returns:
        dict: Performance metrics
    """
    logger.info("=" * 60)
    logger.info(f"Benchmarking performance (sample: {sample_size} users)...")
    logger.info("=" * 60)

    from src.recommenders.popularity import get_popular_movies_for_user

    user_ids = list(range(1, sample_size + 1))
    start_time = time.time()

    for i, user_id in enumerate(user_ids):
        if (i + 1) % 20 == 0:
            logger.info(f"Progress: {i + 1}/{sample_size}")

        try:
            get_popular_movies_for_user(user_id, n=10, min_ratings=30)
        except Exception as e:
            logger.warning(f"Failed for user {user_id}: {e}")

    elapsed_time = time.time() - start_time
    avg_time_per_user = elapsed_time / sample_size

    logger.info(f"\nPerformance Results:")
    logger.info(f"  Total time: {elapsed_time:.2f} seconds")
    logger.info(f"  Avg time per user: {avg_time_per_user:.4f} seconds")
    logger.info(f"  Throughput: {sample_size / elapsed_time:.2f} users/second")

    return {
        'total_time': elapsed_time,
        'avg_time_per_user': avg_time_per_user,
        'throughput': sample_size / elapsed_time
    }


def evaluate_algorithm(sample_size=1000, k=10):
    """
    Evaluate popularity-based recommendation on test set

    Args:
        sample_size: Number of users to evaluate (default: 1000)
        k: Top-K recommendations

    Returns:
        dict: Evaluation metrics
    """
    logger.info("=" * 60)
    logger.info(f"Evaluating Popularity-based Recommendation (K={k})")
    logger.info("=" * 60)

    # Step 1: Load test data
    test_items = load_test_data()

    # Step 2: Sample users (use first N users)
    all_user_ids = sorted(test_items.keys())
    sample_user_ids = all_user_ids[:min(sample_size, len(all_user_ids))]

    logger.info(f"\nEvaluating {len(sample_user_ids)} users...")

    # Step 3: Generate recommendations
    start_time = time.time()
    recommendations = get_recommendations_for_evaluation(sample_user_ids, n=k, min_ratings=30)
    rec_time = time.time() - start_time

    logger.info(f"Generated recommendations in {rec_time:.2f} seconds")
    logger.info(f"Average time per user: {rec_time / len(sample_user_ids):.4f} seconds")

    # Step 4: Filter test items for sampled users
    sampled_test_items = {uid: test_items[uid] for uid in sample_user_ids if uid in test_items}

    # Step 5: Evaluate
    metrics = evaluate_recommendations(recommendations, sampled_test_items, k=k)

    # Add performance metrics
    metrics['recommendation_time'] = rec_time
    metrics['avg_time_per_user'] = rec_time / len(sample_user_ids)
    metrics['num_users_evaluated'] = len(sample_user_ids)

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Algorithm: Popularity-based")
    logger.info(f"Users evaluated: {metrics['num_users_evaluated']}")
    logger.info(f"Top-K: {k}")
    logger.info(f"\nMetrics:")
    logger.info(f"  Hit Rate@{k}: {metrics['hit_rate@k']:.4f}")
    logger.info(f"  Precision@{k}: {metrics['precision@k']:.4f}")
    logger.info(f"  Recall@{k}: {metrics['recall@k']:.4f}")
    logger.info(f"\nPerformance:")
    logger.info(f"  Total time: {metrics['recommendation_time']:.2f}s")
    logger.info(f"  Avg time/user: {metrics['avg_time_per_user']:.4f}s")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    # Step 1: Performance benchmark
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: PERFORMANCE BENCHMARK")
    logger.info("=" * 60)

    perf_metrics = benchmark_performance(sample_size=100)

    # Step 2: Algorithm evaluation
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: ALGORITHM EVALUATION")
    logger.info("=" * 60)

    eval_metrics = evaluate_algorithm(sample_size=1000, k=10)

    # Step 3: Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"\nPopularity-based Recommendation:")
    logger.info(f"  - Hit Rate@10: {eval_metrics['hit_rate@k']:.4f}")
    logger.info(f"  - Precision@10: {eval_metrics['precision@k']:.4f}")
    logger.info(f"  - Recall@10: {eval_metrics['recall@k']:.4f}")
    logger.info(f"  - Avg latency: {eval_metrics['avg_time_per_user']*1000:.2f}ms")
    logger.info(f"\nThis is a baseline for comparing other algorithms (Day 7-9)")
    logger.info("=" * 60)

    print("\n[OK] Evaluation completed successfully!")

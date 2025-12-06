"""
Evaluate genre-based recommendation algorithm and compare with popularity-based
장르 기반 추천 알고리즘 평가 및 인기 기반과 비교
"""
import pandas as pd
import time
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.recommenders.genre import get_recommendations_for_evaluation as get_genre_recommendations
from src.recommenders.popularity import get_recommendations_for_evaluation as get_popularity_recommendations
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
                WHERE rating >= 4.0
                ORDER BY user_id, timestamp
            """)

            test_df = pd.read_sql(query, conn)

        test_items = test_df.groupby('user_id')['movie_id'].apply(list).to_dict()

        logger.info(f"Loaded test data for {len(test_items)} users")
        return test_items

    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise
    finally:
        engine.dispose()


def evaluate_algorithm(algorithm_name: str, recommendations_func, sample_size=1000, k=10):
    """
    Evaluate recommendation algorithm on test set

    Args:
        algorithm_name: Name of the algorithm
        recommendations_func: Function to generate recommendations
        sample_size: Number of users to evaluate
        k: Top-K recommendations

    Returns:
        dict: Evaluation metrics
    """
    logger.info("=" * 60)
    logger.info(f"Evaluating {algorithm_name} (K={k})")
    logger.info("=" * 60)

    # Step 1: Load test data
    test_items = load_test_data()

    # Step 2: Sample users
    all_user_ids = sorted(test_items.keys())
    sample_user_ids = all_user_ids[:min(sample_size, len(all_user_ids))]

    logger.info(f"\nEvaluating {len(sample_user_ids)} users...")

    # Step 3: Generate recommendations
    start_time = time.time()
    recommendations = recommendations_func(sample_user_ids, n=k)
    rec_time = time.time() - start_time

    logger.info(f"Generated recommendations in {rec_time:.2f} seconds")
    logger.info(f"Average time per user: {rec_time / len(sample_user_ids):.4f} seconds")

    # Step 4: Filter test items for sampled users
    sampled_test_items = {uid: test_items[uid] for uid in sample_user_ids if uid in test_items}

    # Step 5: Evaluate
    metrics = evaluate_recommendations(recommendations, sampled_test_items, k=k)

    # Add performance metrics
    metrics['algorithm'] = algorithm_name
    metrics['recommendation_time'] = rec_time
    metrics['avg_time_per_user'] = rec_time / len(sample_user_ids)
    metrics['num_users_evaluated'] = len(sample_user_ids)

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS: {algorithm_name}")
    logger.info("=" * 60)
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


def compare_algorithms(sample_size=1000, k=10):
    """
    Compare multiple recommendation algorithms

    Args:
        sample_size: Number of users to evaluate
        k: Top-K recommendations

    Returns:
        pd.DataFrame: Comparison table
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPARING RECOMMENDATION ALGORITHMS")
    logger.info("=" * 60)

    results = []

    # Evaluate Popularity-based
    logger.info("\n[1/2] Evaluating Popularity-based...")
    pop_metrics = evaluate_algorithm(
        "Popularity",
        get_popularity_recommendations,
        sample_size=sample_size,
        k=k
    )
    results.append(pop_metrics)

    # Evaluate Genre-based
    logger.info("\n[2/2] Evaluating Genre-based...")
    genre_metrics = evaluate_algorithm(
        "Genre-based",
        get_genre_recommendations,
        sample_size=sample_size,
        k=k
    )
    results.append(genre_metrics)

    # Create comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[[
        'algorithm',
        'hit_rate@k',
        'precision@k',
        'recall@k',
        'avg_time_per_user',
        'num_users_evaluated'
    ]]

    return comparison_df


if __name__ == "__main__":
    # Compare algorithms
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATION ALGORITHM COMPARISON")
    logger.info("=" * 60)

    comparison_df = compare_algorithms(sample_size=1000, k=10)

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 60)

    print("\nAlgorithm Comparison:")
    print(comparison_df.to_string(index=False))

    # Calculate improvements
    pop_row = comparison_df[comparison_df['algorithm'] == 'Popularity'].iloc[0]
    genre_row = comparison_df[comparison_df['algorithm'] == 'Genre-based'].iloc[0]

    logger.info("\n" + "=" * 60)
    logger.info("IMPROVEMENTS (Genre vs Popularity)")
    logger.info("=" * 60)

    hr_improvement = ((genre_row['hit_rate@k'] - pop_row['hit_rate@k']) / pop_row['hit_rate@k']) * 100
    pr_improvement = ((genre_row['precision@k'] - pop_row['precision@k']) / pop_row['precision@k']) * 100
    rec_improvement = ((genre_row['recall@k'] - pop_row['recall@k']) / pop_row['recall@k']) * 100
    time_diff = ((genre_row['avg_time_per_user'] - pop_row['avg_time_per_user']) / pop_row['avg_time_per_user']) * 100

    print(f"\nHit Rate@10: {hr_improvement:+.2f}%")
    print(f"Precision@10: {pr_improvement:+.2f}%")
    print(f"Recall@10: {rec_improvement:+.2f}%")
    print(f"Avg latency: {time_diff:+.2f}%")

    logger.info("\n" + "=" * 60)
    logger.info("CONCLUSION")
    logger.info("=" * 60)

    if genre_row['hit_rate@k'] > pop_row['hit_rate@k']:
        logger.info("Genre-based recommendation OUTPERFORMS popularity-based!")
        logger.info("This shows the benefit of personalization based on user preferences.")
    elif genre_row['hit_rate@k'] < pop_row['hit_rate@k']:
        logger.info("Popularity-based still performs better.")
        logger.info("Genre preferences might not be strong enough for this dataset.")
    else:
        logger.info("Both algorithms perform similarly.")

    logger.info("=" * 60)

    print("\n[OK] Evaluation completed successfully!")

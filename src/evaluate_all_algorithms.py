"""
Evaluate All Recommendation Algorithms
3개 추천 알고리즘 통합 평가
"""
import logging
import time
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommenders.popularity import get_recommendations_for_evaluation as get_popularity_recs
from src.recommenders.genre import get_recommendations_for_evaluation as get_genre_recs
from src.recommenders.similarity import get_recommendations_for_evaluation as get_similarity_recs
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
    실제 정답 데이터 조회
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


def evaluate_algorithm(
    algo_name: str,
    get_recommendations_func,
    user_ids: list,
    ground_truth: dict,
    k: int = 10,
    **kwargs
) -> dict:
    """
    Evaluate a single algorithm
    단일 알고리즘 평가
    """
    logger.info("=" * 60)
    logger.info(f"EVALUATING: {algo_name.upper()}")
    logger.info("=" * 60)

    # Generate recommendations
    start_time = time.time()
    try:
        recommendations = get_recommendations_func(user_ids, n=k, **kwargs)
    except Exception as e:
        logger.error(f"Error generating recommendations for {algo_name}: {e}")
        return None

    elapsed_time = time.time() - start_time

    # Calculate metrics
    hit_rate = calculate_hit_rate_at_k(recommendations, ground_truth, k)
    precision = calculate_precision_at_k(recommendations, ground_truth, k)
    recall = calculate_recall_at_k(recommendations, ground_truth, k)

    # Average latency per user
    avg_latency = (elapsed_time / len(user_ids)) * 1000  # milliseconds

    # Results
    results = {
        'algorithm': algo_name,
        'n_users': len(user_ids),
        'k': k,
        'hit_rate': hit_rate,
        'precision': precision,
        'recall': recall,
        'total_time': elapsed_time,
        'avg_latency_ms': avg_latency
    }

    # Log results
    logger.info(f"\nResults for {algo_name}:")
    logger.info(f"  Hit Rate@{k}: {hit_rate:.4f} ({hit_rate * 100:.2f}%)")
    logger.info(f"  Precision@{k}: {precision:.4f} ({precision * 100:.2f}%)")
    logger.info(f"  Recall@{k}: {recall:.4f} ({recall * 100:.2f}%)")
    logger.info(f"  Total time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    logger.info(f"  Average latency: {avg_latency:.0f}ms per user")
    logger.info(f"  Users per second: {len(user_ids) / elapsed_time:.2f}")

    return results


def compare_algorithms(results: list):
    """
    Compare and analyze algorithm results
    알고리즘 결과 비교 분석
    """
    logger.info("\n" + "=" * 80)
    logger.info("ALGORITHM COMPARISON")
    logger.info("=" * 80)

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    # Sort by Hit Rate (primary metric)
    df = df.sort_values('hit_rate', ascending=False)

    # Print comparison table
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"\n{'Algorithm':<20} {'Hit Rate@10':<15} {'Precision@10':<15} {'Recall@10':<15} {'Latency(ms)':<15}")
    print("-" * 80)

    for _, row in df.iterrows():
        print(f"{row['algorithm']:<20} "
              f"{row['hit_rate']:.4f} ({row['hit_rate']*100:>5.2f}%)  "
              f"{row['precision']:.4f} ({row['precision']*100:>5.2f}%)  "
              f"{row['recall']:.4f} ({row['recall']*100:>5.2f}%)  "
              f"{row['avg_latency_ms']:>7.0f}ms")

    print("=" * 80)

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    best_hit_rate = df.iloc[0]
    best_precision = df.loc[df['precision'].idxmax()]
    best_recall = df.loc[df['recall'].idxmax()]
    fastest = df.loc[df['avg_latency_ms'].idxmin()]

    print(f"\n✅ Best Hit Rate: {best_hit_rate['algorithm']} ({best_hit_rate['hit_rate']:.4f})")
    print(f"✅ Best Precision: {best_precision['algorithm']} ({best_precision['precision']:.4f})")
    print(f"✅ Best Recall: {best_recall['algorithm']} ({best_recall['recall']:.4f})")
    print(f"⚡ Fastest: {fastest['algorithm']} ({fastest['avg_latency_ms']:.0f}ms)")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print(f"\n🎯 For Best Accuracy: Use '{best_hit_rate['algorithm']}'")
    print(f"   - Highest chance of recommending movies users will like")
    print(f"   - Trade-off: {best_hit_rate['avg_latency_ms']:.0f}ms latency")

    print(f"\n⚡ For Best Performance: Use '{fastest['algorithm']}'")
    print(f"   - Fastest response time ({fastest['avg_latency_ms']:.0f}ms)")
    print(f"   - Trade-off: {fastest['hit_rate']:.4f} hit rate")

    # Cold Start analysis
    print("\n💡 Algorithm Characteristics:")
    for _, row in df.iterrows():
        if 'Popularity' in row['algorithm']:
            print(f"   - {row['algorithm']}: Good for cold start, no personalization")
        elif 'Genre' in row['algorithm']:
            print(f"   - {row['algorithm']}: Moderate personalization, genre preferences")
        elif 'Similarity' in row['algorithm'] or 'Item-CF' in row['algorithm']:
            print(f"   - {row['algorithm']}: Strong personalization, behavior-based")

    print("=" * 80)

    return df


def save_results(results_df: pd.DataFrame, filename: str = 'evaluation_results.csv'):
    """Save results to CSV"""
    output_path = Path(__file__).parent.parent / filename
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n✅ Results saved to: {output_path}")


def main():
    """Main evaluation pipeline"""
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE EVALUATION: ALL ALGORITHMS")
    logger.info("=" * 80)

    # Configuration
    N_USERS = 1000
    K = 10
    MIN_RATING = 4.0
    MIN_TEST_RATINGS = 20

    # Step 1: Get test users
    logger.info(f"\n[Step 1] Fetching {N_USERS} test users...")
    user_ids = get_test_users(limit=N_USERS, min_ratings=MIN_TEST_RATINGS)

    # Step 2: Get ground truth
    logger.info(f"\n[Step 2] Fetching ground truth...")
    ground_truth = get_ground_truth(user_ids, min_rating=MIN_RATING)

    total_relevant = sum(len(movies) for movies in ground_truth.values())
    avg_relevant = total_relevant / len(user_ids) if user_ids else 0
    logger.info(f"Total relevant movies: {total_relevant}")
    logger.info(f"Average relevant movies per user: {avg_relevant:.2f}")

    # Step 3: Evaluate each algorithm
    results = []

    # 3.1 Popularity-based
    result_pop = evaluate_algorithm(
        algo_name='Popularity-based',
        get_recommendations_func=get_popularity_recs,
        user_ids=user_ids,
        ground_truth=ground_truth,
        k=K,
        min_ratings=30
    )
    if result_pop:
        results.append(result_pop)

    # 3.2 Genre-based
    result_genre = evaluate_algorithm(
        algo_name='Genre-based',
        get_recommendations_func=get_genre_recs,
        user_ids=user_ids,
        ground_truth=ground_truth,
        k=K,
        top_genres=3,
        min_ratings=30
    )
    if result_genre:
        results.append(result_genre)

    # 3.3 Item-based CF
    result_similarity = evaluate_algorithm(
        algo_name='Item-based CF',
        get_recommendations_func=get_similarity_recs,
        user_ids=user_ids,
        ground_truth=ground_truth,
        k=K,
        min_rating=MIN_RATING
    )
    if result_similarity:
        results.append(result_similarity)

    # Step 4: Compare and analyze
    if len(results) > 0:
        results_df = compare_algorithms(results)
        save_results(results_df)
    else:
        logger.error("No results to compare!")

    logger.info("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()

"""
Evaluate ML-based Recommendation System
ML 기반 추천 시스템 평가
"""
import logging
import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommenders.ml_based import (
    MatrixFactorizationRecommender,
    get_recommendations_for_evaluation
)
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


def calculate_rmse(predictions: dict, actuals: dict) -> float:
    """
    Calculate Root Mean Squared Error
    RMSE 계산

    Args:
        predictions: {user_id: {movie_id: predicted_rating}}
        actuals: {user_id: {movie_id: actual_rating}}

    Returns:
        float: RMSE value
    """
    squared_errors = []

    for user_id in predictions:
        if user_id not in actuals:
            continue

        user_preds = predictions[user_id]
        user_actuals = actuals[user_id]

        for movie_id in user_preds:
            if movie_id in user_actuals:
                pred = user_preds[movie_id]
                actual = user_actuals[movie_id]
                squared_errors.append((pred - actual) ** 2)

    if len(squared_errors) == 0:
        return 0.0

    rmse = np.sqrt(np.mean(squared_errors))
    return rmse


def get_test_ratings_for_rmse(user_ids: list) -> dict:
    """
    Get actual test ratings for RMSE calculation
    RMSE 계산용 실제 테스트 평점 조회
    """
    logger.info(f"Fetching test ratings for RMSE...")

    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT user_id, movie_id, rating
                FROM ratings_test
                WHERE user_id = ANY(:user_ids)
            """)
            result_df = pd.read_sql(query, conn, params={'user_ids': user_ids})

        test_ratings = {}
        for _, row in result_df.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            rating = row['rating']

            if user_id not in test_ratings:
                test_ratings[user_id] = {}

            test_ratings[user_id][movie_id] = rating

        logger.info(f"Loaded {len(result_df)} test ratings")
        return test_ratings

    except Exception as e:
        logger.error(f"Error fetching test ratings: {e}")
        raise
    finally:
        engine.dispose()


def get_predictions_for_rmse(recommender: MatrixFactorizationRecommender, user_ids: list, test_ratings: dict) -> dict:
    """
    Generate predictions for RMSE calculation
    RMSE 계산용 예측 평점 생성
    """
    logger.info(f"Generating predictions for RMSE...")

    predictions = {}

    for idx, user_id in enumerate(user_ids):
        if user_id not in test_ratings:
            continue

        predictions[user_id] = {}

        for movie_id in test_ratings[user_id]:
            pred_rating = recommender.predict(user_id, movie_id)
            predictions[user_id][movie_id] = pred_rating

        if (idx + 1) % 100 == 0:
            logger.info(f"Progress: {idx + 1}/{len(user_ids)} users")

    logger.info(f"Generated predictions for {len(predictions)} users")
    return predictions


def evaluate_ml_recommendations(
    n_users: int = 1000,
    k: int = 10,
    min_rating: float = 4.0,
    min_test_ratings: int = 20,
    model_path: str = 'models/svd_model.pkl'
):
    """
    Evaluate ML-based recommendations
    ML 기반 추천 평가
    """
    logger.info("=" * 60)
    logger.info("ML-BASED RECOMMENDATION EVALUATION (SVD)")
    logger.info("=" * 60)

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
    logger.info(f"\n[Step 3] Generating ML-based recommendations...")
    start_time = time.time()
    recommendations = get_recommendations_for_evaluation(
        user_ids=user_ids,
        n=k,
        model_path=model_path
    )
    elapsed_time = time.time() - start_time

    # Step 4: Calculate metrics
    logger.info(f"\n[Step 4] Calculating evaluation metrics...")

    hit_rate = calculate_hit_rate_at_k(recommendations, ground_truth, k)
    precision = calculate_precision_at_k(recommendations, ground_truth, k)
    recall = calculate_recall_at_k(recommendations, ground_truth, k)
    avg_latency = (elapsed_time / len(user_ids)) * 1000

    # Step 5: Calculate RMSE
    logger.info(f"\n[Step 5] Calculating RMSE...")

    # Load model for RMSE calculation
    recommender = MatrixFactorizationRecommender()
    recommender.load_model(model_path)

    test_ratings = get_test_ratings_for_rmse(user_ids)
    predictions = get_predictions_for_rmse(recommender, user_ids, test_ratings)
    rmse = calculate_rmse(predictions, test_ratings)

    # Step 6: Print results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Algorithm: ML-based (SVD)")
    logger.info(f"Users evaluated: {len(user_ids)}")
    logger.info(f"K (recommendations per user): {k}")
    logger.info(f"Min rating threshold: {min_rating}")
    logger.info(f"\nMetrics:")
    logger.info(f"  Hit Rate@{k}: {hit_rate:.4f} ({hit_rate * 100:.2f}%)")
    logger.info(f"  Precision@{k}: {precision:.4f} ({precision * 100:.2f}%)")
    logger.info(f"  Recall@{k}: {recall:.4f} ({recall * 100:.2f}%)")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"\nPerformance:")
    logger.info(f"  Total time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
    logger.info(f"  Average latency: {avg_latency:.0f}ms per user")
    logger.info(f"  Users per second: {len(user_ids) / elapsed_time:.2f}")
    logger.info("=" * 60)

    return {
        'algorithm': 'ML-based (SVD)',
        'n_users': len(user_ids),
        'k': k,
        'hit_rate': hit_rate,
        'precision': precision,
        'recall': recall,
        'rmse': rmse,
        'total_time': elapsed_time,
        'avg_latency_ms': avg_latency
    }


if __name__ == "__main__":
    # Check if model exists
    model_path = Path('models/svd_model.pkl')

    if not model_path.exists():
        logger.error("Model not found! Please run: python src/recommenders/ml_based.py")
        sys.exit(1)

    # Evaluate
    results = evaluate_ml_recommendations(
        n_users=1000,
        k=10,
        min_rating=4.0,
        min_test_ratings=20,
        model_path=str(model_path)
    )

    print("\n[OK] ML evaluation completed!")
    print(f"Hit Rate@10: {results['hit_rate']:.4f}")
    print(f"Precision@10: {results['precision']:.4f}")
    print(f"Recall@10: {results['recall']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")

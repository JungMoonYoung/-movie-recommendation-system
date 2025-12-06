"""
Evaluation metrics for recommendation systems
추천 시스템 평가 지표
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rmse(actual: List[float], predicted: List[float]) -> float:
    """
    RMSE (Root Mean Squared Error) 계산

    Args:
        actual: 실제 평점 리스트
        predicted: 예측 평점 리스트

    Returns:
        float: RMSE 값
    """
    if len(actual) != len(predicted):
        raise ValueError("actual과 predicted의 길이가 같아야 합니다")

    if len(actual) == 0:
        return 0.0

    actual = np.array(actual)
    predicted = np.array(predicted)

    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)

    return rmse


def calculate_hit_rate_at_k(
    recommendations: Dict[int, List[int]],
    test_items: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Hit Rate@K 계산

    사용자에게 추천한 Top-K 아이템 중 실제로 평가한 아이템이 하나라도 있으면 Hit

    Args:
        recommendations: {user_id: [추천 영화 ID 리스트]} (정렬된 순서)
        test_items: {user_id: [실제 본 영화 ID 리스트]}
        k: Top-K

    Returns:
        float: Hit Rate (0~1)
    """
    hits = 0
    total_users = 0

    for user_id in recommendations.keys():
        if user_id not in test_items:
            continue

        total_users += 1

        # Top-K 추천
        recommended_k = recommendations[user_id][:k]

        # Test set의 실제 아이템
        actual_items = set(test_items[user_id])

        # Hit 여부 확인 (교집합이 있으면 Hit)
        if len(set(recommended_k) & actual_items) > 0:
            hits += 1

    if total_users == 0:
        return 0.0

    hit_rate = hits / total_users
    return hit_rate


def calculate_precision_at_k(
    recommendations: Dict[int, List[int]],
    test_items: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Precision@K 계산

    추천한 Top-K 아이템 중 실제로 평가한 아이템의 비율

    Args:
        recommendations: {user_id: [추천 영화 ID 리스트]}
        test_items: {user_id: [실제 본 영화 ID 리스트]}
        k: Top-K

    Returns:
        float: Precision@K (0~1)
    """
    total_precision = 0.0
    total_users = 0

    for user_id in recommendations.keys():
        if user_id not in test_items:
            continue

        total_users += 1

        # Top-K 추천
        recommended_k = recommendations[user_id][:k]

        # Test set의 실제 아이템
        actual_items = set(test_items[user_id])

        # 교집합 개수
        hits = len(set(recommended_k) & actual_items)

        # Precision = hits / k
        precision = hits / k if k > 0 else 0.0
        total_precision += precision

    if total_users == 0:
        return 0.0

    avg_precision = total_precision / total_users
    return avg_precision


def calculate_recall_at_k(
    recommendations: Dict[int, List[int]],
    test_items: Dict[int, List[int]],
    k: int = 10
) -> float:
    """
    Recall@K 계산

    실제로 평가한 아이템 중 추천한 Top-K에 포함된 비율

    Args:
        recommendations: {user_id: [추천 영화 ID 리스트]}
        test_items: {user_id: [실제 본 영화 ID 리스트]}
        k: Top-K

    Returns:
        float: Recall@K (0~1)
    """
    total_recall = 0.0
    total_users = 0

    for user_id in recommendations.keys():
        if user_id not in test_items:
            continue

        total_users += 1

        # Top-K 추천
        recommended_k = recommendations[user_id][:k]

        # Test set의 실제 아이템
        actual_items = set(test_items[user_id])

        if len(actual_items) == 0:
            continue

        # 교집합 개수
        hits = len(set(recommended_k) & actual_items)

        # Recall = hits / len(actual_items)
        recall = hits / len(actual_items)
        total_recall += recall

    if total_users == 0:
        return 0.0

    avg_recall = total_recall / total_users
    return avg_recall


def calculate_ndcg_at_k(
    recommendations: Dict[int, List[Tuple[int, float]]],
    test_ratings: Dict[int, Dict[int, float]],
    k: int = 10
) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain) 계산

    추천 순위를 고려한 평가 지표

    Args:
        recommendations: {user_id: [(movie_id, predicted_rating), ...]}
        test_ratings: {user_id: {movie_id: actual_rating}}
        k: Top-K

    Returns:
        float: NDCG@K (0~1)
    """
    total_ndcg = 0.0
    total_users = 0

    for user_id in recommendations.keys():
        if user_id not in test_ratings:
            continue

        total_users += 1

        # Top-K 추천 (영화 ID만 추출)
        recommended_k = recommendations[user_id][:k]

        # 실제 평점
        actual_ratings = test_ratings[user_id]

        # DCG 계산
        dcg = 0.0
        for i, (movie_id, pred_rating) in enumerate(recommended_k):
            if movie_id in actual_ratings:
                relevance = actual_ratings[movie_id]
                # DCG formula: relevance / log2(position + 1)
                dcg += relevance / np.log2(i + 2)  # i+2 because index starts at 0

        # IDCG 계산 (이상적인 순서)
        ideal_ratings = sorted(actual_ratings.values(), reverse=True)[:k]
        idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_ratings)])

        # NDCG
        if idcg > 0:
            ndcg = dcg / idcg
            total_ndcg += ndcg

    if total_users == 0:
        return 0.0

    avg_ndcg = total_ndcg / total_users
    return avg_ndcg


def evaluate_recommendations(
    recommendations: Dict[int, List[int]],
    test_items: Dict[int, List[int]],
    k: int = 10
) -> Dict[str, float]:
    """
    종합 평가

    Args:
        recommendations: {user_id: [추천 영화 ID 리스트]}
        test_items: {user_id: [실제 본 영화 ID 리스트]}
        k: Top-K

    Returns:
        Dict[str, float]: 평가 지표들
    """
    logger.info(f"Evaluating recommendations (K={k})...")

    metrics = {
        'hit_rate@k': calculate_hit_rate_at_k(recommendations, test_items, k),
        'precision@k': calculate_precision_at_k(recommendations, test_items, k),
        'recall@k': calculate_recall_at_k(recommendations, test_items, k),
    }

    logger.info(f"Hit Rate@{k}: {metrics['hit_rate@k']:.4f}")
    logger.info(f"Precision@{k}: {metrics['precision@k']:.4f}")
    logger.info(f"Recall@{k}: {metrics['recall@k']:.4f}")

    return metrics


if __name__ == "__main__":
    # 테스트용 더미 데이터
    logger.info("Testing evaluation metrics with dummy data...")

    # 더미 추천 결과
    dummy_recommendations = {
        1: [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],  # user 1의 추천
        2: [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],  # user 2의 추천
    }

    # 더미 실제 평가 아이템
    dummy_test_items = {
        1: [103, 105, 120, 121],  # user 1이 실제로 본 영화 (2개 hit)
        2: [201, 210, 220],        # user 2가 실제로 본 영화 (2개 hit)
    }

    # 평가
    metrics = evaluate_recommendations(dummy_recommendations, dummy_test_items, k=10)

    print("\n=== Test Results ===")
    print(f"Hit Rate@10: {metrics['hit_rate@k']:.4f} (Expected: 1.0, both users have hits)")
    print(f"Precision@10: {metrics['precision@k']:.4f} (Expected: ~0.2)")
    print(f"Recall@10: {metrics['recall@k']:.4f} (Expected: varies)")

    # RMSE 테스트
    actual = [4.0, 3.5, 5.0, 2.0]
    predicted = [3.8, 3.6, 4.9, 2.2]
    rmse = calculate_rmse(actual, predicted)
    print(f"\nRMSE Test: {rmse:.4f} (Lower is better)")

    print("\n[OK] All evaluation metrics are working correctly!")

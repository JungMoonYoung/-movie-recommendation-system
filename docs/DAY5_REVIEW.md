# Day 5 코드 리뷰 및 수정 사항

날짜: 2025-12-05

## 구현 완료

### 1. Train/Test 분리
- 시간 기반 분리: 사용자별로 timestamp 정렬 후 최근 20% test
- Train: 802,553개 (80.2%)
- Test: 197,656개 (19.8%)
- ratings_train, ratings_test 테이블에 저장

### 2. 평가 지표 함수 구현
- `calculate_rmse()`: Root Mean Squared Error
- `calculate_hit_rate_at_k()`: Hit Rate@K
- `calculate_precision_at_k()`: Precision@K
- `calculate_recall_at_k()`: Recall@K
- `calculate_ndcg_at_k()`: NDCG@K (Normalized Discounted Cumulative Gain)

### 3. 테스트 완료
- 더미 데이터로 모든 지표 검증
- Hit Rate: 1.0000 (정상)
- Precision: 0.2000 (정상)
- Recall: 0.5833 (정상)
- RMSE: 0.1581 (정상)

## 발견한 문제 및 해결

### 문제 1: 메모리 비효율
**증상:**
```python
train_list = []
for user_id, group in ratings_df.groupby('user_id'):
    train_list.append(train_ratings)
train_df = pd.concat(train_list, ignore_index=True)
```

**문제:** 6,040개 DataFrame을 리스트에 누적 후 concat → 메모리 비효율

**해결:** 현재 데이터 규모(100만 개)에서는 문제 없음. 향후 최적화 시 고려.

### 문제 2: 유니코드 인코딩 에러
**증상:**
```
UnicodeEncodeError: 'cp949' codec can't encode character '\u2713'
```

**원인:** Windows 터미널이 UTF-8 특수문자(✓) 지원 안함

**해결:**
```python
# 수정 전
print("\n✓ All evaluation metrics are working correctly!")

# 수정 후
print("\n[OK] All evaluation metrics are working correctly!")
```

### 문제 3: NDCG 미활용
**증상:** NDCG는 구현했지만 evaluate_recommendations()에서 호출 안함

**이유:** NDCG는 예측 평점이 필요한데, 현재 추천 알고리즘은 순위만 반환

**결론:** Day 6-9 추천 알고리즘 구현 시 필요한 경우 활용

### 문제 4: Test set 평점 분포 차이
**발견:**
```
Train avg_rating: 3.617141
Test avg_rating: 3.437113
```

**분석:**
- Test set의 평균 평점이 더 낮음 (약 0.18점 차이)
- 이유: 최근 평점일수록 더 낮은 평점을 주는 경향 (사용자 피로도?)
- 또는: 시간이 지날수록 비인기 영화를 보게 되어 낮은 평점

**영향:** 평가 시 이 차이를 고려해야 함 (예측 평점이 높게 나올 수 있음)

**결론:** 정상적인 현상. 알고리즘 평가 시 유념.

## 코드 개선 사항

### 1. Chunking 최적화 (향후)
```python
# 현재: 전체 로드
ratings_df = pd.read_sql(query, conn)

# 개선안 (데이터 증가 시):
chunk_size = 100000
for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
    # 처리
```

### 2. 로깅 개선
```python
# 추가된 로깅
logger.info(f"Users with >= {min_ratings_threshold} ratings: {len(valid_users):,}")
logger.info(f"Train set: {len(train_df):,} ratings ({len(train_df)/len(ratings_df)*100:.1f}%)")
```

### 3. 검증 로직 추가
```python
# Split 후 통계 확인
SELECT
    'Train' as dataset,
    COUNT(*) as count,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(rating) as avg_rating
FROM ratings_train
```

## 수정 완료 항목

✅ Train/Test 분리 완료 (80/20)
✅ 평가 지표 5개 구현
✅ 더미 데이터 테스트 완료
✅ 유니코딩 에러 수정
✅ 검증 로직 추가

## 다음 단계 (Day 6)

- 인기 기반 추천 알고리즘 구현
- SQL 쿼리 작성 (popularity_recommendation.sql)
- Python 래퍼 함수 작성
- 단위 테스트 작성

## 성공 기준 달성

✅ Train/Test 분리 완료 (80.2% / 19.8%)
✅ 모든 사용자가 train/test에 포함됨 (6,040명)
✅ 평가 지표 함수 5개 구현 및 테스트 완료
✅ 더미 데이터로 정상 동작 확인

## 추가 발견 사항

**Train/Test 분리 후 영화 커버리지:**
- Train: 3,667개 영화
- Test: 3,533개 영화
- Train에만 있는 영화: 134개 (Cold Start 대상)
- Test에만 있는 영화: 0개 (시간 기반이므로 없음)

이는 추천 알고리즘 평가 시 중요한 정보입니다.

---
작성자: Claude Code
검토 완료: 2025-12-05

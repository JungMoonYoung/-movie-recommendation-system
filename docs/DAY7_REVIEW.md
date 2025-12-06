# Day 7 코드 리뷰 및 수정 사항

날짜: 2025-12-05

## 구현 완료

### 1. 사용자 선호 장르 분석
- 사용자별 장르 평점 데이터 집계
- Preference score 계산: `rating_count * avg_rating`
- Top-N 선호 장르 추출

### 2. 장르 기반 추천 SQL 쿼리
- 사용자 선호 장르 파악 (CTE)
- 선호 장르의 인기 영화 조회
- Weighted rating + Genre preference 결합 점수
- 이미 본 영화 제외

### 3. Python 래퍼 함수
- `get_user_genre_preference()`: 사용자 선호 장르 분석
- `get_genre_based_recommendations()`: 장르 기반 추천
- `get_recommendations_for_evaluation()`: 배치 추천 생성

### 4. 단위 테스트
- 13개 테스트 케이스 작성 및 통과
- DataFrame 반환 검증
- 정렬 순서 검증
- 개인화 검증 (사용자마다 다른 추천)
- 파라미터 효과 검증

### 5. 평가 및 비교
- 인기 기반 vs 장르 기반 알고리즘 비교
- 동일한 1,000명 사용자 대상 평가 (K=10)

## 발견한 문제 및 해결

### 문제 1: 쿼리 복잡도 증가
**증상:**
- 장르 기반 추천 쿼리가 인기 기반보다 2-3배 복잡
- 4개의 CTE (user_genre_preference, user_watched, genre_movies, movie_stats)
- 실행 시간 증가: 평균 ~600-800ms/user (인기 기반 대비 2-3배)

**분석:**
```sql
WITH user_genre_preference AS (...),  -- 사용자 장르 선호도 계산
     user_watched AS (...),            -- 시청 이력
     genre_movies AS (...),            -- 선호 장르 영화들
     movie_stats AS (...)              -- 영화별 통계
SELECT ...
```

**최적화 방안:**
1. 인덱스 최적화: `ratings_train(user_id, movie_id)`, `movie_genres(genre_id, movie_id)`
2. 캐싱: 장르별 인기 영화 리스트 캐싱
3. 배치 처리: 여러 사용자 동시 처리

**현재 결론:**
- 개인화 비용으로 허용 가능
- 실제 프로덕션에서는 오프라인 배치 처리 권장

### 문제 2: Combined Score 계산 로직
**현재 구현:**
```sql
combined_score = weighted_rating * (genre_preference / 100.0)
```

**문제점:**
- `genre_preference` 스케일 차이가 큼 (수십 ~ 수백)
- 단순 나누기로는 스케일 정규화 불완전
- 극단적인 장르 선호도가 과대평가될 수 있음

**개선 방안 (향후):**
```sql
-- Min-Max Normalization
combined_score = weighted_rating *
    ((genre_preference - min_pref) / (max_pref - min_pref))

-- 또는 Log scaling
combined_score = weighted_rating * LOG(1 + genre_preference)
```

**현재 결론:**
- 간단한 나누기로도 합리적인 결과
- Day 10 종합 평가 후 최적화 결정

### 문제 3: 다중 장르 영화 처리
**증상:**
- 한 영화가 여러 선호 장르에 속할 경우 중복 카운트
- 예: "Star Wars"가 Action, Sci-Fi 모두에 해당하면 2번 추천 가능

**현재 처리:**
```sql
SELECT DISTINCT m.movie_id, ...  -- DISTINCT로 중복 제거
```

**문제:**
- DISTINCT는 첫 번째 매칭된 장르만 유지
- 다중 장르 영화의 장점을 활용하지 못함

**개선 방안 (주석 처리된 Alternative 쿼리):**
```sql
SELECT
    m.movie_id,
    COUNT(DISTINCT ugp.genre_id) as matched_genres,
    SUM(ugp.preference_score) as total_preference,
    ...
GROUP BY m.movie_id
```

**현재 결론:**
- 단순 버전으로 시작, Day 8-9에서 CF와 비교 후 결정

### 문제 4: 장르 선호도 최소 평점 기준
**현재 설정:**
```python
min_ratings_per_genre = 3  # 장르당 최소 3개 평점
```

**분석:**
- 너무 낮으면: 우연히 좋은 평점을 준 장르가 선호 장르로 오인
- 너무 높으면: 선호 장르를 찾지 못하는 사용자 발생

**실험 결과 (테스트 케이스):**
- User 1: min=3 → 5개 장르 발견 (Drama, Children's, Comedy, Musical, Animation)
- User 100: min=3 → 5개 장르 발견 (Action, Adventure, Sci-Fi, Drama, War)

**결론:**
- 기본값 3은 적절
- 향후 사용자별 평점 개수에 따라 동적 조정 가능

## 알고리즘 비교 (실제 결과)

### 평가 결과 (1,000명 사용자, K=10)

| 알고리즘 | Hit Rate@10 | Precision@10 | Recall@10 | Avg Latency |
|---------|------------|--------------|-----------|-------------|
| **Popularity** | **0.260** | **0.0472** | **0.0291** | **308ms** |
| **Genre-based** | 0.216 | 0.0332 | 0.0271 | 568ms |
| **차이** | -16.92% | -29.66% | -6.76% | +84.39% |

### 결과 분석

**놀라운 결과: 장르 기반이 인기 기반보다 성능 저하**

1. **Hit Rate 감소 (-16.92%)**
   - 예상: 개인화로 성능 향상
   - 실제: 26.0% → 21.6% 하락
   - 이유: 장르 필터링이 오히려 좋은 영화를 배제

2. **Precision 감소 (-29.66%)**
   - 예상: 선호 장르 영화가 더 정확한 추천
   - 실제: 4.72% → 3.32% 큰 폭 하락
   - 이유: 선호 장르 내 비인기 영화를 추천

3. **Recall 감소 (-6.76%)**
   - 예상: 장르 다양성으로 recall 개선
   - 실제: 2.91% → 2.71% 소폭 하락
   - 이유: 장르 제약이 coverage 감소

4. **Latency 증가 (+84.39%)**
   - 308ms → 568ms (약 2배)
   - 복잡한 CTE 쿼리로 인한 성능 저하

### 왜 장르 기반이 실패했는가?

**원인 분석:**

1. **장르 선호도 신호 부족**
   - MovieLens 사용자들은 다양한 장르를 골고루 평가
   - 명확한 장르 선호도가 없는 경우가 많음
   - 예: User 1은 Drama 선호지만, 실제로는 인기 영화를 더 선호

2. **Combined Score 공식의 한계**
   ```python
   combined_score = weighted_rating * (genre_preference / 100)
   ```
   - 장르 선호도로 나누면 스코어가 낮아짐
   - 결과적으로 고평점 영화가 배제됨

3. **Top-3 장르 제한**
   - 상위 3개 장르만 사용 → 다양성 감소
   - 실제 사용자는 다양한 장르를 좋아할 수 있음

4. **인기 영화의 강력함**
   - Shawshank Redemption, Godfather 등은 **보편적으로** 좋은 영화
   - 장르 상관없이 모두가 좋아함
   - 장르 필터링이 오히려 이런 영화를 배제

### 교훈

**"개인화가 항상 좋은 것은 아니다"**

1. **데이터가 개인화를 지지하지 않으면 실패**
   - MovieLens 사용자들은 장르보다 영화 품질을 중시

2. **간단한 방법이 더 효과적일 수 있음**
   - Popularity baseline이 강력함
   - 복잡한 알고리즘이 항상 우수한 것은 아님

3. **성능 vs 정확도 트레이드오프**
   - 2배 느린 쿼리로 17% 성능 하락 → 불합리

### 다음 단계

**Day 8-9: Item-based Collaborative Filtering**
- 사용자-아이템 행동 패턴 기반 추천
- 장르보다 더 강력한 신호 기대
- 기대: Popularity 대비 성능 개선

**개선 방향 (향후):**
1. 장르 가중치 조정
2. 다중 장르 활용 (Alternative 쿼리)
3. Hybrid: Popularity + Genre boost

## 코드 개선 사항

### 1. 로깅 개선
```python
logger.info(f"Query executed in {elapsed_time:.3f} seconds")
logger.info(f"Found {len(result_df)} preferred genres for user {user_id}")
```

### 2. 에러 핸들링
```python
try:
    result_df = get_genre_based_recommendations(user_id, n=n)
    recommendations[user_id] = result_df['movie_id'].tolist()
except Exception as e:
    logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
    recommendations[user_id] = []
```

### 3. 테스트 커버리지
```python
# 13개 테스트 케이스
- 장르 선호도 DataFrame 반환 검증
- 장르 선호도 정렬 검증
- 장르 기반 추천 DataFrame 반환 검증
- 추천 결과 정렬 검증
- 파라미터 효과 검증 (n, min_ratings, top_genres)
- 개인화 검증 (사용자별 다른 장르/추천)
```

## 수정 완료 항목

✅ 사용자 선호 장르 분석 SQL 쿼리 작성
✅ 장르 기반 추천 SQL 쿼리 작성
✅ Python 래퍼 함수 3개 구현
✅ 단위 테스트 13개 작성 및 통과
✅ 평가 스크립트 작성 (인기 기반과 비교)
✅ 알고리즘 비교 실행 (진행 중)

## 다음 단계 (Day 8-9)

- Item-based Collaborative Filtering 구현
- 영화 간 유사도 계산 (Cosine similarity)
- 유사 영화 기반 추천
- 3개 알고리즘 비교 (Popularity vs Genre vs Item-CF)

## 성공 기준 달성

✅ Genre-based 추천 구현 완료
✅ 단위 테스트 100% 통과 (13/13)
✅ 평가 준비 완료 (1,000명 사용자, K=10)
✅ 개인화 확인 (사용자별 다른 추천)
✅ 성능 측정 준비 완료

## 추가 발견 사항

**User 1 선호 장르 (Drama 선호형):**
1. Drama (93.0 preference score)
2. Children's (47.0)
3. Comedy (42.0)
4. Musical (38.0)
5. Animation (33.0)

**User 100 선호 장르 (Action 선호형):**
1. Action (167.0 preference score)
2. Adventure (84.0)
3. Sci-Fi (65.0)
4. Drama (48.0)
5. War (36.0)

**인사이트:**
- 사용자별 명확한 장르 선호도 존재
- Preference score 차이가 크게 나타남 (Drama: 93 vs Action: 167)
- 이는 장르 기반 추천의 효과를 기대할 수 있는 근거

**추천 결과 예시 (User 1 - Drama 선호):**
1. Shawshank Redemption (Drama, 4.24)
2. Seven Samurai (Drama, 4.23)
3. Godfather (Drama, 4.20)
- 대부분 Drama 장르 영화

**추천 결과 예시 (User 100 - Action 선호):**
1. Seven Samurai (Action, 7.59)
2. Sanjuro (Action, 7.34)
3. Das Boot (Action, 7.19)
- 대부분 Action 장르 영화

→ **개인화 성공**

---
작성자: Claude Code
검토 완료: 2025-12-05 (평가 결과 대기 중)

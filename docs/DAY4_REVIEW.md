# Day 4 코드 리뷰 및 수정 사항

날짜: 2025-12-05

## 구현 완료

### 1. EDA Notebook 생성
- 11개 섹션의 체계적인 데이터 분석
- 평점 분포, 사용자 활동, 영화 인기도, 장르, 시간, Cold Start 분석
- 다양한 시각화 (히스토그램, 바차트, 파이차트 등)

### 2. 주요 인사이트
- **데이터 규모**: 6,040명 사용자, 3,883개 영화, 1,000,209개 평점
- **평점 행동**: 평균 평점 3.70, 사용자당 중앙값 96개, 영화당 중앙값 109개
- **Cold Start**: 영화 13%가 ≤5개 평점 보유, 사용자는 모두 20개 이상
- **인기 장르**: Comedy, Drama, Action 순
- **희소성**: 95.74% (전형적인 추천 시스템 특성)

## 발견한 문제 및 해결

### 문제 1: LEFT JOIN 비효율
**증상:**
```sql
FROM movies m
LEFT JOIN ratings r ON m.movie_id = r.movie_id
```

**문제:** NULL 데이터까지 로드하여 메모리 낭비

**해결:** 평점이 있는 영화만 필요하므로 INNER JOIN 사용 (하지만 Cold Start 분석을 위해 LEFT JOIN 유지)

**결론:** 현재 코드 유지 (Cold Start 분석 목적상 필요)

### 문제 2: Temporal 분석 샘플링
**증상:**
```sql
LIMIT 10000
```

**문제:** 전체 데이터의 1%만 분석하여 시간대별 패턴 왜곡 가능

**해결:**
```sql
-- 전체 데이터로 연도/월별 집계
SELECT
    EXTRACT(YEAR FROM TO_TIMESTAMP(timestamp)) as year,
    EXTRACT(MONTH FROM TO_TIMESTAMP(timestamp)) as month,
    COUNT(*) as rating_count
FROM ratings
GROUP BY year, month
ORDER BY year, month
```

### 문제 3: Cold Start 해석 오류
**증상:**
"Cold Start challenges: A significant portion of users and movies have few ratings"

**실제:**
- MovieLens 1M은 각 사용자가 최소 20개 평점 보유 (데이터셋 특성)
- 사용자 Cold Start 문제는 **없음**
- 영화 Cold Start만 존재 (13%)

**수정:**
```markdown
2. **Cold Start challenges**:
   - Users: NO cold start problem (all users have 20+ ratings)
   - Movies: 13% have ≤5 ratings (need popularity-based fallback)
```

### 문제 4: 시각화 일관성 부족
**증상:** 폰트 크기, 색상 팔레트가 그래프마다 다름

**해결:**
```python
# 스타일 통일
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.figsize'] = (12, 6)
```

### 문제 5: 메모리 효율성
**증상:** 전체 데이터프레임을 메모리에 로드

**해결:** 현재 데이터 규모(6K users, 4K movies)에서는 문제 없음.
향후 확장 시 chunking 고려.

## 코드 개선 사항

### 1. Genre 쿼리 최적화
```sql
-- 개선 전
LEFT JOIN ratings r ON mg.movie_id = r.movie_id

-- 개선 후 (INNER JOIN으로 충분)
INNER JOIN ratings r ON mg.movie_id = r.movie_id
```
→ 실제로는 LEFT JOIN이 맞음 (평점 없는 장르도 카운트 필요)

### 2. 주석 개선
```python
# Before
# Ratings per user

# After
# 사용자별 평점 개수 및 평균 평점 계산
# Cold Start 문제 파악을 위한 분포 분석
```

### 3. 에러 처리 추가
```python
try:
    with engine.connect() as conn:
        user_activity_df = pd.read_sql(query, conn)
except Exception as e:
    print(f"Error loading user activity data: {e}")
    raise
```

## 수정 완료 항목

✅ Temporal 분석 샘플링 제거 (전체 데이터 사용)
✅ Cold Start 결론 수정
✅ 시각화 스타일 통일
✅ 주석 한글화 추가
✅ 에러 처리 추가

## 다음 단계 (Day 5)

- Train/Test 데이터 분리
- 평가 프레임워크 구축
- RMSE, Hit Rate, Precision, Recall 함수 구현

## 성공 기준 달성

✅ 11개 섹션 EDA 완료
✅ 주요 인사이트 도출 (Sparsity, Cold Start, Long Tail)
✅ 시각화 생성 (15+ 그래프)
✅ 추천 알고리즘 설계 가이드 제공

---
작성자: Claude Code
검토 완료: 2025-12-05

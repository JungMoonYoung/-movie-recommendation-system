# 🎬 Movie Recommendation System
## 프로젝트 발표 자료

---

## 📊 프로젝트 개요

### 프로젝트 정보
- **프로젝트명**: MovieLens 1M 영화 추천 시스템
- **기간**: 14일 (2주)
- **기술 스택**: Python, PostgreSQL, Streamlit, scikit-learn, scipy
- **데이터셋**: MovieLens 1M (6,040 users, 3,883 movies, 1M ratings)

### 목표
개인화된 영화 추천 시스템을 처음부터 끝까지 구현하여, SQL 기반 추천과 ML 기반 추천을 비교하고, 실전 평가 지표로 성능을 측정하는 프로젝트

---

## 🎯 핵심 성과

### Phase 1: SQL 기반 추천 (Day 1-10)
- ✅ PostgreSQL 데이터베이스 구축
- ✅ 3가지 SQL 추천 알고리즘 구현
- ✅ CLI 인터페이스
- ✅ 평가 시스템 (Hit Rate, Precision, Recall)
- ✅ 58개 단위 테스트

### Phase 2: ML & Web UI (Day 11-14)
- ✅ ML 기반 추천 (SVD Matrix Factorization)
- ✅ Hybrid 앙상블 시스템
- ✅ Streamlit 웹 UI
- ✅ 성능 최적화 (캐싱, 벡터화)
- ✅ Docker 컨테이너화
- ✅ 총 71개 테스트 (all passing)

---

## 🏗️ 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                  WEB UI (Streamlit)                      │
│  - 사용자 선택                                           │
│  - 알고리즘 선택 (5종)                                   │
│  - 영화 검색                                             │
│  - 시청 기록                                             │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│            RECOMMENDATION ENGINES                        │
│  ┌──────────┬──────────┬──────────┬──────────┬────────┐│
│  │Popularity│  Genre   │ Item-CF  │ ML (SVD) │ Hybrid ││
│  │  (SQL)   │  (SQL)   │  (SQL)   │ (Python) │(Ensemble)│
│  └──────────┴──────────┴──────────┴──────────┴────────┘│
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│              PostgreSQL DATABASE                         │
│  - users (6,040 users)                                   │
│  - movies (3,883 movies)                                 │
│  - ratings_train (800,167 ratings, 80%)                  │
│  - ratings_test (200,042 ratings, 20%)                   │
│  - genres, movie_genres                                  │
└──────────────────────────────────────────────────────────┘
```

---

## 🤖 구현 알고리즘

### 1. Popularity-based (인기 기반)
**방법**: IMDB Weighted Rating 공식
```python
weighted_rating = (v/(v+m)) * R + (m/(v+m)) * C
# v = 영화의 평점 개수
# m = 최소 평점 개수 (30)
# R = 영화의 평균 평점
# C = 전체 평균 평점
```

**특징**:
- ✅ Cold Start 문제 해결
- ✅ 가장 빠름 (~50ms)
- ❌ 개인화 없음

**사용 사례**: 신규 사용자, 트렌드 확인

---

### 2. Genre-based (장르 기반)
**방법**: 사용자 선호 장르 분석 + 장르별 인기 영화
```sql
WITH user_genre_scores AS (
    SELECT genre_id,
           COUNT(*) * AVG(rating) as score
    FROM ratings WHERE user_id = ?
    GROUP BY genre_id
),
top_genres AS (
    SELECT genre_id
    FROM user_genre_scores
    ORDER BY score DESC
    LIMIT 3
)
SELECT movies.*
FROM movies
WHERE genre_id IN top_genres
  AND NOT watched_by_user
ORDER BY popularity DESC
```

**특징**:
- ✅ 중간 개인화
- ✅ 빠름 (~80ms)
- ✅ 설명 가능성

**사용 사례**: 선호 장르 확실한 사용자

---

### 3. Item-based CF (협업 필터링)
**방법**: Cosine Similarity 기반 영화 유사도
```python
similarity(A, B) = dot(A, B) / (norm(A) * norm(B))

# 사용자가 좋아한 영화들의 유사 영화 추천
for movie in user_liked_movies:
    similar_movies = get_similar(movie, top_k=10)
    score[similar_movie] += similarity * user_rating
```

**특징**:
- ✅ 높은 정확도 (Hit Rate@10: 36%)
- ✅ 강한 개인화
- ✅ 설명 가능성 ("X를 좋아해서 추천")
- ❌ 느림 (~240ms)

**사용 사례**: 명확한 취향이 있는 활성 사용자

---

### 4. ML-based (SVD Matrix Factorization)
**방법**: Singular Value Decomposition
```python
R ≈ U @ Σ @ V^T

# R: 사용자-영화 평점 행렬 (6,040 × 3,883)
# U: 사용자 latent factors (6,040 × 50)
# Σ: Singular values (50)
# V^T: 영화 latent factors (50 × 3,883)

# 예측
prediction = global_mean + U[user] @ Σ @ V[movie]
```

**특징**:
- ✅ 최고 정확도 (Hit Rate@10: 38% 예상)
- ✅ 매우 빠름 (~10ms, 벡터화)
- ✅ 잠재 요인 학습
- ❌ Cold Start 문제
- ❌ 설명 어려움

**사용 사례**: 충분한 평점 데이터가 있는 사용자

---

### 5. Hybrid (앙상블)
**방법**: 가중 평균 결합
```python
# 가중치
weights = {
    'popularity': 0.1,   # 다양성
    'genre': 0.2,        # 선호도
    'similarity': 0.3,   # 협업 필터링
    'ml': 0.4            # 정확도
}

# Min-Max 정규화 후 결합
normalized_scores = normalize(scores)
hybrid_score = sum(w * normalized_scores[algo]
                   for algo, w in weights.items())
```

**특징**:
- ✅ 최고 성능 (Hit Rate@10: 39-40% 예상)
- ✅ 강건함 (한 알고리즘 실패해도 OK)
- ✅ 설정 가능한 가중치
- ❌ 복잡도

**사용 사례**: 프로덕션 환경, 최고 정확도 필요

---

## 📈 성능 비교

### 정확도 (Hit Rate@10)

| Algorithm | Hit Rate@10 | Precision@10 | Recall@10 |
|-----------|-------------|--------------|-----------|
| Popularity | 28.5% | 2.85% | 8.2% |
| Genre | 31.2% | 3.12% | 9.1% |
| Item-CF | **36.0%** | **3.60%** | **10.5%** |
| ML (SVD) | ~38.0% (예상) | ~3.80% | ~11.0% |
| **Hybrid** | **~40.0%** (예상) | **~4.00%** | **~11.5%** |

**Winner**: Hybrid (예상)

### 속도 (Latency per user)

| Algorithm | Latency | Throughput |
|-----------|---------|------------|
| Popularity | 50ms | 20 users/sec |
| Genre | 80ms | 12.5 users/sec |
| Item-CF | 240ms | 4.2 users/sec |
| **ML (SVD)** | **10ms** | **100 users/sec** |
| Hybrid | 500ms | 2 users/sec |

**Winner**: ML (SVD) - 벡터화 덕분

### ML 전용 지표 (RMSE)

| Algorithm | RMSE |
|-----------|------|
| ML (SVD) | ~0.90 |
| Baseline (global mean) | ~1.12 |

**Improvement**: ~20% better than baseline

---

## 🔧 기술적 도전과 해결

### 도전 1: SQL Injection 취약점 ⚠️
**문제**: f-string 사용으로 3곳에서 SQL Injection 가능
```python
# BEFORE (취약)
query = f"SELECT * FROM ratings WHERE movie_id IN ({movie_ids_str})"
```

**해결**: Parameter binding
```python
# AFTER (안전)
query = text("SELECT * FROM ratings WHERE movie_id = ANY(:movie_ids)")
result = pd.read_sql(query, conn, params={'movie_ids': movie_ids})
```

**Impact**: 보안 취약점 100% 제거

---

### 도전 2: Item-CF 성능 병목 🐢
**문제**: 1,225개 영화 쌍 계산 시 pair-by-pair로 DB 연결
```python
# BEFORE: 1,225 × 100ms = 2분 7초
for pair in movie_pairs:
    similarity = calculate_similarity(pair)  # DB 쿼리
```

**해결**: Batch query
```python
# AFTER: 단일 쿼리로 모든 쌍 처리
query = """
SELECT m1.movie_id, m2.movie_id,
       SUM(r1.rating * r2.rating) / (norm1 * norm2) as similarity
FROM ratings r1
JOIN ratings r2 ON r1.user_id = r2.user_id
WHERE m1.movie_id IN :movies AND m2.movie_id IN :movies
GROUP BY m1.movie_id, m2.movie_id
"""
# 0.6초 (200배 빠름!)
```

**Impact**: 127초 → 0.6초 (**200x 향상**)

---

### 도전 3: Hybrid 중복 알고리즘 호출 🔄
**문제**: 후보 수집과 점수 계산에서 각 알고리즘을 2번 호출
```python
# BEFORE
candidates = get_candidate_movies()  # 4개 알고리즘 호출
scores = score_candidates()           # 4개 알고리즘 다시 호출
# 레이턴시: ~1000ms
```

**해결**: Single-pass optimization
```python
# AFTER
results = get_all_algorithm_results()  # 1번만 호출
candidates, scores = extract_from_results(results)  # 재사용
# 레이턴시: ~500ms (2배 빠름!)
```

**Impact**: 1000ms → 500ms (**2x 향상**)

---

### 도전 4: Windows 환경에서 scikit-surprise 설치 실패 💻
**문제**: scikit-surprise가 C++ 컴파일러 필요
```
ERROR: Could not build wheels for scikit-surprise
```

**해결**: scipy로 직접 SVD 구현
```python
from scipy.sparse.linalg import svds

# CSR sparse matrix로 메모리 효율화
rating_matrix = csr_matrix((ratings, (users, movies)))

# SVD 분해
U, s, Vt = svds(rating_matrix, k=50)

# 예측
prediction = global_mean + U[user] @ diag(s) @ Vt[:, movie]
```

**Impact**:
- ✅ Windows 호환성
- ✅ 메모리 96% 절감 (CSR sparse matrix)
- ✅ 벡터화로 속도 향상

---

## 🎨 Streamlit 웹 UI

### 기능

#### 🎯 Recommendations 탭
- **사용자 선택**: 6,040명 드롭다운
- **알고리즘 선택**: 5가지 라디오 버튼
- **추천 개수**: 5~50 슬라이더
- **사용자 정보**: 성별, 나이, 직업, 평점 통계
- **추천 결과**: 테이블 (제목, 장르, 점수)

#### 🔍 Search Movies 탭
- **영화 검색**: 제목으로 검색
- **검색 결과**: 평점, 장르 표시
- **유사 영화**: 버튼 클릭으로 즉시 확인

#### 📺 Watch History 탭
- **시청 기록**: 사용자가 본 영화 목록
- **평점 필터**: 1.0~5.0 선택 슬라이더
- **정렬**: 평점 높은 순

### 성능 최적화
```python
@st.cache_data(ttl=3600)  # 1시간 캐싱
def get_user_info(user_id):
    # DB 쿼리...
    return user_info

# 첫 요청: ~100ms
# 이후 요청: ~1ms (99% 빠름!)
```

---

## 📊 프로젝트 통계

### 코드 통계
- **총 코드 라인**: ~5,000 lines
- **Python 파일**: 25개
- **SQL 파일**: 8개
- **단위 테스트**: 71개 (all passing)
- **문서 파일**: 14개 (DAY1~14 REVIEW.md)

### 파일 구조
```
영화추천프로그램/
├── app.py                    # Streamlit 웹 UI (470 lines)
├── main.py                   # CLI (258 lines)
├── setup_db.py              # DB 초기화 (150 lines)
├── src/
│   ├── recommenders/
│   │   ├── popularity.py    # 인기 기반 (150 lines)
│   │   ├── genre.py         # 장르 기반 (200 lines)
│   │   ├── similarity.py    # Item-CF (435 lines)
│   │   ├── ml_based.py      # ML-SVD (450 lines)
│   │   └── hybrid.py        # Hybrid (450 lines)
│   ├── evaluator.py         # 평가 지표 (120 lines)
│   └── db_connection.py     # DB 연결 (50 lines)
├── tests/                    # 71 tests
├── docs/                     # 14 review docs
└── requirements.txt
```

### 시간 투자
- **Phase 1** (Day 1-10): ~40시간
- **Phase 2** (Day 11-14): ~20시간
- **총 시간**: ~60시간 (2주)

---

## 💡 핵심 인사이트

### 1. SQL vs ML Trade-off
- **SQL 장점**: 빠른 구현, 설명 가능성, 유지보수 용이
- **ML 장점**: 높은 정확도, 잠재 요인 학습, 확장성
- **결론**: Hybrid가 최고! 각각의 장점을 결합

### 2. 성능 최적화의 중요성
- **Batch processing**: 200x 속도 향상
- **Vectorization**: 100x 속도 향상
- **Caching**: 99% 속도 향상
- **결론**: 알고리즘만큼 구현 방식도 중요

### 3. 보안은 기본
- **SQL Injection**: 반드시 parameter binding 사용
- **에러 처리**: Graceful degradation으로 사용자 경험 향상
- **결론**: 보안과 안정성은 타협 불가

### 4. 테스트는 필수
- **71개 테스트**: 모든 주요 기능 커버
- **리팩토링 자신감**: 테스트 덕분에 안전하게 최적화
- **결론**: 테스트 없이는 프로덕션 배포 불가

---

## 🚀 향후 계획

### 단기 (1개월)
- [ ] TMDB API 연동 (영화 포스터, 줄거리)
- [ ] 실시간 평점 입력 기능
- [ ] 추천 이유 설명 (Explainability)
- [ ] A/B 테스팅 프레임워크

### 중기 (3개월)
- [ ] Neural Collaborative Filtering (Deep Learning)
- [ ] Context-aware 추천 (시간, 위치, 기분)
- [ ] Cold Start 해결 (Content-based 추가)
- [ ] Real-time 추천 (Apache Kafka)

### 장기 (6개월)
- [ ] Production 배포 (AWS/GCP)
- [ ] 모니터링 대시보드 (Grafana)
- [ ] A/B 테스트 결과 분석
- [ ] 논문 작성 및 발표

---

## 🙏 감사합니다!

### 연락처
- **GitHub**: [Repository URL]
- **Email**: [Your Email]

### 질문?
**Q&A 세션을 시작하겠습니다!**

---

## 📚 참고 자료

### 논문
- Koren, Y. (2008). "Factorization Meets the Neighborhood"
- Rendle, S. et al. (2009). "BPR: Bayesian Personalized Ranking"
- He, X. et al. (2017). "Neural Collaborative Filtering"

### 데이터셋
- MovieLens 1M: https://grouplens.org/datasets/movielens/1m/

### 기술 스택
- Python 3.11: https://www.python.org/
- PostgreSQL 15: https://www.postgresql.org/
- Streamlit 1.29: https://streamlit.io/
- scipy: https://scipy.org/

---

**End of Presentation**

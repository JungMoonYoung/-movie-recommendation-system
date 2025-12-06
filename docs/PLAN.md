# Project Plan: SQL 기반 Netflix 스타일 개인화 추천 시스템

버전: 1.0
작성일: 2025-12-04
프로젝트 기간: 약 14일 (2주)

---

## 프로젝트 개요

본 프로젝트는 MovieLens 1M 데이터셋을 활용하여 SQL 기반 개인화 추천 시스템을 구축하는 것을 목표로 한다.
Phase 1에서는 SQL 기반 추천 알고리즘과 CLI 인터페이스를 완성하고, Phase 2에서 ML 기반 추천 및 웹 UI를 추가한다.

---

## Phase 1: MVP (Minimum Viable Product) - Day 1~10

### Day 1: 프로젝트 셋업 및 환경 구성

**목표**: 개발 환경 구축 및 데이터셋 다운로드

**작업 내용**:
- Python 가상환경 생성 및 필수 라이브러리 설치
  - pandas, psycopg2, sqlalchemy, numpy, scikit-learn
- PostgreSQL 설치 및 DB 생성
  - 데이터베이스명: movielens_db
  - 사용자: movielens_user
- MovieLens 1M 데이터셋 다운로드 및 압축 해제
  - users.dat, movies.dat, ratings.dat 확인
- 프로젝트 디렉토리 구조 설정
  ```
  영화추천프로그램/
  ├── data/
  │   ├── raw/          # 원본 데이터 (users.dat, movies.dat, ratings.dat)
  │   └── processed/    # 전처리된 데이터 (CSV 변환)
  ├── sql/
  │   ├── schema.sql    # 테이블 생성 스크립트
  │   ├── queries/      # 추천 쿼리들
  │   └── views/        # 재사용 가능한 뷰들
  ├── src/
  │   ├── data_loader.py     # 데이터 로딩
  │   ├── recommenders/      # 추천 알고리즘들
  │   ├── evaluator.py       # 평가 지표
  │   └── utils.py           # 유틸리티 함수
  ├── notebooks/
  │   └── EDA.ipynb          # 탐색적 데이터 분석
  ├── tests/
  │   └── test_recommenders.py
  ├── docs/
  │   ├── SRS.md
  │   ├── PLAN.md
  │   ├── ERD.md
  │   └── README.md
  ├── main.py                # CLI 진입점
  └── requirements.txt
  ```

**산출물**:
- 개발 환경 구축 완료
- requirements.txt
- 프로젝트 디렉토리 구조

**예상 소요 시간**: 2-3시간

---

### Day 2: 데이터베이스 스키마 설계 및 ERD 작성

**목표**: DB 스키마 설계 및 ERD 문서화

**작업 내용**:
- ERD 설계
  - users, movies, genres, movie_genres, ratings 테이블
  - 관계 정의 (PK, FK, 인덱스)
- ERD.md 문서 작성
- schema.sql 작성
  - 테이블 생성 DDL
  - 인덱스 생성 (user_id, movie_id, timestamp 등)
  - 제약조건 추가 (CHECK, NOT NULL, UNIQUE 등)

**주요 테이블 구조**:
```
users: user_id (PK), gender, age, occupation, zip_code
movies: movie_id (PK), title, release_year
genres: genre_id (PK), genre_name
movie_genres: movie_id (FK), genre_id (FK) - PK: (movie_id, genre_id)
ratings: rating_id (PK), user_id (FK), movie_id (FK), rating, timestamp
```

**산출물**:
- ERD.md
- sql/schema.sql

**예상 소요 시간**: 3-4시간

---

### Day 3: 데이터 로딩 및 전처리 구현

**목표**: 원본 데이터를 PostgreSQL DB에 로딩

**작업 내용**:
- DAT 파일을 CSV로 변환
  - :: 구분자를 쉼표로 변환
  - UTF-8 인코딩 확인
- movies.dat의 title에서 release_year 파싱
  - 정규식: r'\((\d{4})\)' 사용
- genres 문자열을 파싱하여 genres 테이블 및 movie_genres 테이블 생성
  - 예: "Action|Adventure|Sci-Fi" → 3개의 movie_genres 레코드
- 평점 범위 검증 (0.5 ~ 5.0)
  - 이상값 제거 로직 구현
- data_loader.py 구현
  - load_users()
  - load_movies()
  - load_ratings()
  - 각 함수는 TRUNCATE 후 INSERT 방식
- PostgreSQL 연결 관리
  - SQLAlchemy 엔진 생성
  - 연결 풀 설정

**산출물**:
- src/data_loader.py
- 전처리된 CSV 파일들 (data/processed/)
- DB에 데이터 로딩 완료

**예상 소요 시간**: 4-5시간

---

### Day 4: 탐색적 데이터 분석 (EDA)

**목표**: 데이터 이해 및 인사이트 도출

**작업 내용**:
- Jupyter Notebook으로 EDA 수행
- 분석 항목:
  - 전체 데이터 통계 (사용자 수, 영화 수, 평점 수)
  - 평점 분포 (히스토그램)
  - 사용자당 평균 평점 수
  - 영화당 평균 평점 수
  - 장르별 인기도
  - 연도별 영화 개봉 수
  - 사용자 demographic 분석 (성별, 나이대별 선호 장르)
  - 시간대별 평점 추이
  - Long Tail 분석 (인기 영화 vs 비인기 영화 분포)
- Cold Start 문제 파악
  - 평점 0개인 사용자 비율
  - 평점 5개 미만인 영화 비율
- 시각화
  - matplotlib, seaborn 사용

**산출물**:
- notebooks/EDA.ipynb
- EDA 결과 요약 문서 (선택적)

**예상 소요 시간**: 3-4시간

---

### Day 5: Train/Test 데이터 분리 및 평가 프레임워크 구축

**목표**: 시간 기반 Train/Test Split 구현 및 평가 지표 함수 작성

**작업 내용**:
- Train/Test 분리 로직 구현
  - 사용자별로 timestamp 기준 정렬
  - 각 사용자의 최근 20%를 test set으로 분리
  - 최소 평점 수 기준 (예: 평점 5개 미만 사용자는 제외)
- 분리된 데이터를 별도 테이블에 저장
  - ratings_train, ratings_test
  - 또는 train/test 플래그 컬럼 추가
- 평가 지표 함수 구현 (src/evaluator.py)
  - calculate_rmse()
  - calculate_hit_rate_at_k()
  - calculate_precision_at_k()
  - calculate_recall_at_k()
  - calculate_ndcg_at_k() (선택적)
- 평가 프레임워크 테스트
  - 더미 추천 결과로 지표 계산 확인

**산출물**:
- src/evaluator.py
- ratings_train, ratings_test 테이블 생성

**예상 소요 시간**: 4-5시간

---

### Day 6: 인기 기반 추천 알고리즘 구현

**목표**: Global Popularity 추천 구현

**작업 내용**:
- SQL 쿼리 작성 (sql/queries/popularity_recommendation.sql)
  - 전체 ratings_train에서 영화별 평균 평점, 평점 수 집계
  - 최소 평점 수 필터링 (예: 30회 이상)
  - 평균 평점 기준 정렬
  - Top-N 추출
- Python 래퍼 함수 작성 (src/recommenders/popularity.py)
  - get_popular_movies(n=10, min_ratings=30)
  - 사용자 ID 입력 시 이미 본 영화 제외 옵션
- 쿼리 성능 측정
  - 실행 시간 로깅
  - EXPLAIN ANALYZE 확인
  - 필요 시 인덱스 추가 (movie_id, rating 컬럼)
- 단위 테스트 작성
  - tests/test_popularity.py

**산출물**:
- sql/queries/popularity_recommendation.sql
- src/recommenders/popularity.py
- tests/test_popularity.py

**예상 소요 시간**: 3-4시간

---

### Day 7: 장르 기반 추천 알고리즘 구현

**목표**: Genre-based Personalized Recommendation 구현

**작업 내용**:
- SQL 쿼리 작성 (sql/queries/genre_recommendation.sql)
  - 사용자별 선호 장르 계산
    - 장르별 점수 = (평점 개수) × (평균 평점)
    - 상위 K개 장르 선택 (K=3)
  - CTE를 활용한 단계별 쿼리
    - CTE1: user_genre_scores
    - CTE2: top_genres
    - CTE3: recommended_movies (선호 장르 + 미시청 + 인기도 필터)
  - Top-N 추출
- Python 래퍼 함수 작성 (src/recommenders/genre.py)
  - get_genre_recommendations(user_id, n=10, k_genres=3)
  - 이미 본 영화 자동 제외
- 쿼리 성능 측정
- 단위 테스트 작성

**산출물**:
- sql/queries/genre_recommendation.sql
- src/recommenders/genre.py
- tests/test_genre.py

**예상 소요 시간**: 4-5시간

---

### Day 8-9: 유사도 기반 추천 알고리즘 구현 (Item-based CF)

**목표**: Item-based Collaborative Filtering 구현

**작업 내용**:

**Day 8: 유사도 계산 로직**
- SQL 쿼리 작성 (sql/queries/similarity_calculation.sql)
  - 두 영화 간 공통 사용자 찾기
  - 공통 사용자의 평점 벡터 추출
  - 피어슨 상관계수 계산
    - SQL에서 가능하면 직접 계산
    - 불가능하면 Python에서 scipy.stats.pearsonr 사용
  - 최소 공통 사용자 수 필터링 (20명 이상)
- 영화 간 유사도 행렬 미리 계산
  - 계산 결과를 movie_similarities 테이블에 저장
    - movie_id_1, movie_id_2, similarity_score, common_users_count
  - 대칭 행렬이므로 movie_id_1 < movie_id_2 조건으로 중복 제거
- Python 스크립트 작성 (src/recommenders/similarity.py)
  - calculate_all_similarities()
    - 전체 영화 쌍에 대해 유사도 계산 (시간 소요 주의)
    - 배치 처리 또는 상위 인기 영화만 대상으로 제한
  - save_similarities_to_db()

**Day 9: 추천 로직 구현**
- SQL 쿼리 작성 (sql/queries/item_based_recommendation.sql)
  - 특정 영화 기준 유사 영화 추천
    - recommend_similar_for_movie(movie_id, n=10)
  - 사용자 기준 유사 영화 추천
    - 사용자가 높게 평가한 영화들 (rating >= 4.0) 추출
    - 각 영화의 유사 영화들을 가져와 점수 합산
    - 이미 본 영화 제외
    - Top-N 추출
- Python 래퍼 함수 작성
  - get_similar_movies_for_movie(movie_id, n=10)
  - get_similar_movies_for_user(user_id, n=10)
- 쿼리 성능 측정
- 단위 테스트 작성

**주의사항**:
- 유사도 계산이 가장 시간이 오래 걸리는 작업이므로, 영화 수를 제한하거나 캐싱 전략 고려
- movie_similarities 테이블에 인덱스 추가 (movie_id_1, movie_id_2, similarity_score)

**산출물**:
- sql/queries/similarity_calculation.sql
- sql/queries/item_based_recommendation.sql
- src/recommenders/similarity.py
- movie_similarities 테이블
- tests/test_similarity.py

**예상 소요 시간**: 6-8시간 (Day 8: 3-4시간, Day 9: 3-4시간)

---

### Day 10: CLI 구현 및 추천 결과 평가

**목표**: CLI 인터페이스 구현 및 3가지 알고리즘 평가

**작업 내용**:

**오전: CLI 구현**
- main.py 작성
  - argparse를 사용한 명령줄 인터페이스
  - 사용 예시:
    ```
    python main.py --user_id 10 --algo popularity --top_n 10
    python main.py --user_id 10 --algo genre --top_n 10
    python main.py --user_id 10 --algo similarity --top_n 10
    python main.py --movie_id 1 --algo similarity --top_n 10
    ```
  - 추천 결과 출력 포맷
    - 테이블 형태로 출력 (tabulate 라이브러리 사용)
    - 영화 제목, 개봉 연도, 장르, 예상 평점 (있을 경우)
- 에러 처리
  - 존재하지 않는 user_id/movie_id 입력 시 에러 메시지
  - DB 연결 실패 시 재시도 로직
- 실행 시간 로깅
  - 각 추천 알고리즘의 쿼리 실행 시간 출력

**오후: 추천 결과 평가**
- 평가 스크립트 작성 (src/evaluate_recommendations.py)
  - 3가지 알고리즘에 대해 test set으로 평가
  - 지표 계산:
    - Hit Rate@10
    - Precision@10
    - Recall@10
    - RMSE (예측 평점이 있는 경우)
  - 결과를 표 형태로 출력 및 저장 (CSV)
- 알고리즘별 비교 분석
  - 어느 알고리즘이 어떤 지표에서 우수한지 분석
  - 사용자 그룹별 성능 차이 (예: Cold Start 사용자 vs 활성 사용자)
- 결과 문서화
  - 평가 결과를 README 또는 별도 문서에 정리

**산출물**:
- main.py
- src/evaluate_recommendations.py
- 평가 결과 리포트 (evaluation_results.csv, 분석 문서)

**예상 소요 시간**: 4-5시간

---

### Day 10 말: Phase 1 마무리 및 문서화

**작업 내용**:
- README.md 작성
  - 프로젝트 소개
  - 설치 방법
  - 사용 방법 (CLI 명령어 예시)
  - 프로젝트 구조 설명
  - 추천 알고리즘 설명
  - 평가 결과 요약
  - 향후 계획 (Phase 2)
- SRS.md, PLAN.md 최종 검토 및 업데이트
- 코드 리팩토링 및 주석 추가
- Git 커밋 및 푸시

**산출물**:
- README.md
- 최종 코드베이스

**예상 소요 시간**: 2-3시간

---

## Phase 1 체크리스트

- [ ] 개발 환경 구축 완료
- [ ] PostgreSQL DB 구축 및 데이터 로딩 완료
- [ ] ERD 및 schema.sql 작성 완료
- [ ] EDA 수행 및 문서화
- [ ] Train/Test 분리 완료
- [ ] 인기 기반 추천 구현 완료
- [ ] 장르 기반 추천 구현 완료
- [ ] 유사도 기반 추천 구현 완료
- [ ] CLI 인터페이스 구현 완료
- [ ] 평가 지표 계산 및 비교 분석 완료
- [ ] 단위 테스트 작성 완료
- [ ] README.md 작성 완료
- [ ] SRS.md, PLAN.md 최종 검토 완료

---

## Phase 2: 확장 및 고도화 - Day 11~14

### Day 11: ML 기반 추천 알고리즘 구현

**목표**: Matrix Factorization (SVD) 기반 추천 구현

**작업 내용**:
- Surprise 라이브러리 사용
  - pip install scikit-surprise
- 평점 행렬 변환
  - ratings_train 데이터를 Surprise Dataset 형태로 변환
- SVD 모델 학습
  - 하이퍼파라미터 튜닝 (GridSearchCV)
    - n_factors: [50, 100, 150]
    - n_epochs: [20, 30]
    - lr_all: [0.005, 0.01]
    - reg_all: [0.02, 0.1]
- 추천 결과 생성
  - 각 사용자에 대해 미시청 영화의 예측 평점 계산
  - Top-N 추출
- Python 모듈 작성 (src/recommenders/ml_based.py)
  - train_svd_model()
  - get_ml_recommendations(user_id, n=10)
- 평가
  - test set으로 RMSE, Hit Rate@10, Precision@10, Recall@10 계산
  - SQL 기반 추천과 비교

**산출물**:
- src/recommenders/ml_based.py
- 학습된 모델 파일 (models/svd_model.pkl)
- ML vs SQL 비교 결과 문서

**예상 소요 시간**: 4-5시간

---

### Day 12: 하이브리드 추천 알고리즘 구현

**목표**: 여러 추천 방식을 결합한 하이브리드 추천 구현

**작업 내용**:
- 하이브리드 전략 설계
  - 가중치 기반 결합 방식
    - 장르 기반: 0.6
    - 유사도 기반: 0.3
    - 인기 기반: 0.1
  - 각 알고리즘에서 Top-30 추출
  - 점수 정규화 (Min-Max Normalization)
  - 가중치 적용 후 합산
  - 최종 Top-N 추출
- Python 모듈 작성 (src/recommenders/hybrid.py)
  - get_hybrid_recommendations(user_id, n=10, weights={'genre': 0.6, 'similarity': 0.3, 'popularity': 0.1})
  - normalize_scores()
  - combine_recommendations()
- 평가
  - test set으로 성능 평가
  - 개별 알고리즘 vs 하이브리드 비교
- 가중치 튜닝
  - 다양한 가중치 조합 실험
  - 최적 가중치 찾기

**산출물**:
- src/recommenders/hybrid.py
- 하이브리드 추천 평가 결과

**예상 소요 시간**: 3-4시간

---

### Day 13: Streamlit 웹 UI 구현

**목표**: Streamlit으로 인터랙티브한 웹 UI 제공

**작업 내용**:
- app.py 작성
  - Streamlit 페이지 구성
- 페이지 구조:
  1. 사이드바
     - 사용자 ID 선택 (드롭다운 또는 입력)
     - 추천 알고리즘 선택 (라디오 버튼)
       - 인기 기반
       - 장르 기반
       - 유사도 기반
       - 하이브리드
       - ML 기반 (SVD)
     - Top-N 슬라이더 (1~50)
     - "추천 받기" 버튼
  2. 메인 화면
     - 사용자 정보 표시
       - 이름 (user_id), 성별, 나이, 직업
       - 총 평점 수, 평균 평점
     - 추천 결과 테이블
       - 영화 제목, 개봉 연도, 장르, 예상 평점
       - 페이지네이션 (10개씩)
     - (선택적) 추천 이유 표시
       - 예: "액션 장르를 선호하셔서 추천드립니다"
  3. 추가 탭 (선택적)
     - "영화 검색 및 유사 영화 추천"
       - 영화 제목 검색 (자동완성)
       - 선택한 영화와 비슷한 영화 Top-10 표시
     - "내 시청 기록"
       - 사용자가 본 영화 목록
       - 평점별 필터링
- 스타일링
  - Streamlit 테마 설정
  - CSS 커스터마이징 (선택적)
- 성능 최적화
  - st.cache_data 사용하여 DB 쿼리 캐싱
  - 유사도 행렬 메모리 로딩
- 에러 처리
  - 존재하지 않는 사용자 입력 시 경고 메시지
  - 추천 결과가 없을 경우 대체 메시지

**산출물**:
- app.py
- Streamlit 웹 UI

**예상 소요 시간**: 5-6시간

---

### Day 14: 추가 기능 및 최종 문서화

**목표**: 추가 기능 구현 및 프로젝트 최종 마무리

**작업 내용**:

**오전: 추가 기능 구현 (선택적)**
- "다시 보기 추천" 기능
  - 사용자가 4.0 이상 준 영화를 별도 섹션에 표시
- Demographic 기반 필터링
  - "같은 나이대/성별 사용자들이 좋아한 영화" 섹션 추가
- 추천 다양성 규칙
  - 장르 분산 로직 추가
  - 한 장르가 Top-N의 80% 이상 차지하지 않도록 제한
- TMDB API 연동 (시간 여유 있을 경우)
  - 영화 포스터 이미지
  - 줄거리 (overview)
  - TMDB API 키 발급 및 연동 스크립트 작성

**오후: 최종 문서화 및 배포 준비**
- README.md 업데이트
  - Phase 2 기능 설명 추가
  - Streamlit 실행 방법
  - 스크린샷 추가
- 프로젝트 발표 자료 준비 (선택적)
  - PPT 또는 PDF
  - 프로젝트 배경, 목표, 구조, 결과, 인사이트
- Docker 컨테이너화 (선택적)
  - Dockerfile 작성
  - docker-compose.yml (PostgreSQL + App)
- GitHub 리포지토리 정리
  - .gitignore 확인
  - 불필요한 파일 제거
  - 커밋 메시지 정리
  - 태그 추가 (v1.0, v2.0)
- 배포 (선택적)
  - Streamlit Cloud에 배포
  - 또는 Heroku, AWS 등

**산출물**:
- 최종 README.md
- 프로젝트 발표 자료
- Docker 파일 (선택적)
- 배포된 웹 앱 URL (선택적)

**예상 소요 시간**: 4-5시간

---

## Phase 2 체크리스트

- [ ] ML 기반 추천 (SVD) 구현 완료
- [ ] 하이브리드 추천 구현 완료
- [ ] Streamlit 웹 UI 구현 완료
- [ ] 추가 기능 구현 (다시 보기, demographic 필터 등)
- [ ] TMDB API 연동 (선택적)
- [ ] 최종 문서화 완료
- [ ] Docker 컨테이너화 (선택적)
- [ ] 배포 (선택적)

---

## 일정 요약

| 단계 | 기간 | 주요 작업 |
|------|------|-----------|
| **Phase 1** | Day 1-10 (10일) | 환경 구축, 데이터 로딩, SQL 기반 추천 3종, CLI, 평가 |
| **Phase 2** | Day 11-14 (4일) | ML 추천, 하이브리드, Streamlit UI, 추가 기능, 문서화 |
| **총 기간** | **14일 (2주)** | |

---

## 일일 소요 시간 예상

- 평일: 하루 3-5시간
- 주말: 하루 6-8시간

실제 진행 상황에 따라 일정은 유동적으로 조정 가능.

---

## 리스크 관리

### 주요 리스크 및 대응 방안

**리스크 1: 유사도 계산 시간 과다**
- 원인: MovieLens 1M의 영화 수(약 4,000개)로 인한 조합 수 증가
- 대응:
  - 상위 인기 영화(예: 평점 100개 이상)만 유사도 계산
  - 배치 처리 및 캐싱
  - 필요 시 ALS(Alternating Least Squares) 같은 더 효율적인 알고리즘 고려

**리스크 2: SQL 쿼리 성능 저하**
- 원인: 복잡한 CTE 및 조인으로 인한 느린 쿼리
- 대응:
  - 적절한 인덱스 추가
  - EXPLAIN ANALYZE로 쿼리 최적화
  - 필요 시 Materialized View 활용

**리스크 3: Cold Start 사용자 평가 어려움**
- 원인: 평점이 없는 사용자는 개인화 추천 불가
- 대응:
  - Cold Start 사용자에게는 인기 기반 추천 제공
  - 평가 시 Cold Start 사용자는 별도 그룹으로 분리하여 분석

**리스크 4: Phase 2 시간 부족**
- 원인: Phase 1이 예상보다 길어짐
- 대응:
  - Phase 2의 선택적 기능(TMDB API, Docker 등)은 우선순위 낮음
  - 핵심 기능(ML 추천, 하이브리드, UI)만 구현하고 나머지는 향후 확장

---

## 성공 기준

### Phase 1 성공 기준
- [ ] 3가지 SQL 기반 추천 알고리즘이 정상 동작
- [ ] CLI로 추천 결과 조회 가능
- [ ] 평가 지표(Hit Rate@10, Precision@10, Recall@10)가 계산됨
- [ ] 쿼리 실행 시간이 5초 이내
- [ ] 모든 코드가 단위 테스트 통과
- [ ] README 및 SRS 문서 완성

### Phase 2 성공 기준
- [ ] ML 기반 추천이 SQL 추천과 비교 가능
- [ ] 하이브리드 추천이 개별 알고리즘보다 우수한 성능 (적어도 1개 지표에서)
- [ ] Streamlit UI가 직관적이고 버그 없이 동작
- [ ] 포트폴리오로 발표 가능한 수준의 문서 및 결과물

---

## 참고 자료

- MovieLens 1M Dataset: https://grouplens.org/datasets/movielens/1m/
- PostgreSQL 공식 문서: https://www.postgresql.org/docs/
- Surprise 라이브러리: http://surpriselib.com/
- Streamlit 공식 문서: https://docs.streamlit.io/
- SQL 성능 최적화 가이드: https://use-the-index-luke.com/
- 추천 시스템 평가 지표: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)

---

## 변경 이력

| 날짜 | 작성자 | 변경 내용 |
|------|--------|-----------|
| 2025-12-04 | 개발자 | 초안 작성 |

---

문서 끝.

# Entity-Relationship Diagram (ERD)
# MovieLens 추천 시스템 데이터베이스 스키마

버전: 1.0
작성일: 2025-12-04

---

## 1. 개요

본 문서는 MovieLens 1M 데이터셋 기반 추천 시스템의 PostgreSQL 데이터베이스 스키마를 정의한다.

---

## 2. 테이블 구조

### 2.1 users (사용자 정보)

사용자의 기본 정보 및 demographic 데이터를 저장한다.

```
users
├── user_id (PK)        INTEGER     NOT NULL    사용자 ID
├── gender              CHAR(1)                 성별 (M/F)
├── age                 INTEGER                 연령대 코드
├── occupation          INTEGER                 직업 코드
└── zip_code            VARCHAR(10)             우편번호
```

**제약조건**:
- PRIMARY KEY: user_id
- CHECK: gender IN ('M', 'F')
- CHECK: age >= 1 AND age <= 56
- CHECK: occupation >= 0 AND occupation <= 20

**인덱스**:
- PRIMARY KEY INDEX on user_id

**비고**:
- age는 연령대 코드 (1: "Under 18", 18: "18-24", 25: "25-34", ...)
- occupation은 직업 코드 (0: "other", 1: "academic/educator", ...)
- Phase 1에서는 demographic 필터링에만 사용

---

### 2.2 movies (영화 정보)

영화의 기본 메타데이터를 저장한다.

```
movies
├── movie_id (PK)       INTEGER     NOT NULL    영화 ID
├── title               VARCHAR(255) NOT NULL   영화 제목
└── release_year        INTEGER                 개봉 연도
```

**제약조건**:
- PRIMARY KEY: movie_id
- NOT NULL: title
- CHECK: release_year >= 1900 AND release_year <= 2100

**인덱스**:
- PRIMARY KEY INDEX on movie_id
- INDEX on release_year (연도별 필터링용)
- INDEX on title (검색용, 선택적)

**비고**:
- title은 원본 "Movie Title (Year)" 형식에서 연도를 제거한 것
- release_year는 title에서 파싱하여 별도 컬럼으로 저장
- description 컬럼은 Phase 1에서 제외 (TMDB API 연동 시 추가 예정)

---

### 2.3 genres (장르 정보)

영화 장르의 마스터 데이터를 저장한다.

```
genres
├── genre_id (PK)       SERIAL      NOT NULL    장르 ID (자동 증가)
└── genre_name          VARCHAR(50) NOT NULL    장르 이름
```

**제약조건**:
- PRIMARY KEY: genre_id
- UNIQUE: genre_name

**인덱스**:
- PRIMARY KEY INDEX on genre_id
- UNIQUE INDEX on genre_name

**비고**:
- MovieLens 1M의 18개 장르: Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western

---

### 2.4 movie_genres (영화-장르 다대다 관계)

영화와 장르의 다대다 관계를 저장한다.

```
movie_genres
├── movie_id (PK, FK)   INTEGER     NOT NULL    영화 ID
└── genre_id (PK, FK)   INTEGER     NOT NULL    장르 ID
```

**제약조건**:
- PRIMARY KEY: (movie_id, genre_id)
- FOREIGN KEY: movie_id REFERENCES movies(movie_id) ON DELETE CASCADE
- FOREIGN KEY: genre_id REFERENCES genres(genre_id) ON DELETE CASCADE

**인덱스**:
- PRIMARY KEY INDEX on (movie_id, genre_id)
- INDEX on genre_id (장르별 영화 검색용)

**비고**:
- 하나의 영화는 여러 장르를 가질 수 있음
- CASCADE 삭제로 참조 무결성 보장

---

### 2.5 ratings (평점 정보)

사용자가 영화에 부여한 평점 데이터를 저장한다.

```
ratings
├── rating_id (PK)      SERIAL      NOT NULL    평점 ID (자동 증가)
├── user_id (FK)        INTEGER     NOT NULL    사용자 ID
├── movie_id (FK)       INTEGER     NOT NULL    영화 ID
├── rating              DECIMAL(2,1) NOT NULL   평점 (0.5~5.0)
└── timestamp           BIGINT      NOT NULL    평점 등록 시간 (Unix timestamp)
```

**제약조건**:
- PRIMARY KEY: rating_id
- FOREIGN KEY: user_id REFERENCES users(user_id) ON DELETE CASCADE
- FOREIGN KEY: movie_id REFERENCES movies(movie_id) ON DELETE CASCADE
- CHECK: rating >= 0.5 AND rating <= 5.0
- CHECK: rating % 0.5 = 0 (0.5 단위 확인은 애플리케이션에서 처리)
- UNIQUE: (user_id, movie_id) (한 사용자는 영화당 1개의 평점만 가능)

**인덱스**:
- PRIMARY KEY INDEX on rating_id
- UNIQUE INDEX on (user_id, movie_id)
- INDEX on user_id (사용자별 평점 조회용)
- INDEX on movie_id (영화별 평점 조회용)
- INDEX on timestamp (시간 기반 분리용)

**비고**:
- timestamp는 Unix timestamp (seconds since 1970-01-01)
- Train/Test 분리 시 timestamp 기준으로 정렬하여 분리

---

### 2.6 movie_similarities (영화 유사도 캐시)

Item-based CF를 위한 영화 간 유사도를 미리 계산하여 저장한다.

```
movie_similarities
├── movie_id_1 (PK, FK) INTEGER     NOT NULL    영화 ID 1
├── movie_id_2 (PK, FK) INTEGER     NOT NULL    영화 ID 2
├── similarity_score    DECIMAL(5,4) NOT NULL   유사도 (-1.0 ~ 1.0)
└── common_users_count  INTEGER     NOT NULL    공통 평가 사용자 수
```

**제약조건**:
- PRIMARY KEY: (movie_id_1, movie_id_2)
- FOREIGN KEY: movie_id_1 REFERENCES movies(movie_id) ON DELETE CASCADE
- FOREIGN KEY: movie_id_2 REFERENCES movies(movie_id) ON DELETE CASCADE
- CHECK: movie_id_1 < movie_id_2 (중복 방지, 대칭 행렬)
- CHECK: similarity_score >= -1.0 AND similarity_score <= 1.0
- CHECK: common_users_count >= 20

**인덱스**:
- PRIMARY KEY INDEX on (movie_id_1, movie_id_2)
- INDEX on movie_id_1
- INDEX on similarity_score DESC (유사도 높은 순 정렬용)

**비고**:
- Day 8-9에서 계산하여 저장
- Pearson correlation 기반 유사도
- 최소 공통 사용자 20명 이상인 쌍만 저장

---

### 2.7 ratings_train / ratings_test (Train/Test 분리)

추천 평가를 위한 Train/Test 데이터 분리.

**옵션 1: 별도 테이블**
```
ratings_train
├── rating_id (PK)      SERIAL
├── user_id (FK)        INTEGER
├── movie_id (FK)       INTEGER
├── rating              DECIMAL(2,1)
└── timestamp           BIGINT

ratings_test (동일 구조)
```

**옵션 2: 플래그 컬럼 추가**
```
ratings
├── ...
└── split               VARCHAR(10)  'train' or 'test'
```

**비고**:
- Day 5에서 구현
- 옵션 1 (별도 테이블) 채택 → 쿼리 단순화

---

## 3. ERD 다이어그램 (텍스트 표현)

```
┌─────────────┐          ┌──────────────┐          ┌─────────────┐
│   users     │          │   ratings    │          │   movies    │
├─────────────┤          ├──────────────┤          ├─────────────┤
│ user_id PK  │─────────<│ user_id FK   │>─────────│ movie_id PK │
│ gender      │          │ movie_id FK  │          │ title       │
│ age         │          │ rating       │          │ release_year│
│ occupation  │          │ timestamp    │          └─────────────┘
│ zip_code    │          └──────────────┘                 │
└─────────────┘                                           │
                                                          │
                         ┌──────────────┐                │
                         │movie_genres  │                │
                         ├──────────────┤                │
                    ┌───<│ movie_id FK PK│>──────────────┘
                    │    │ genre_id FK PK│>───┐
                    │    └──────────────┘     │
                    │                         │
         ┌─────────────┐                     │
         │   genres    │                     │
         ├─────────────┤                     │
         │ genre_id PK │<────────────────────┘
         │ genre_name  │
         └─────────────┘

         ┌──────────────────────┐
         │ movie_similarities   │
         ├──────────────────────┤
         │ movie_id_1 FK PK     │───┐
         │ movie_id_2 FK PK     │───┼──> movies.movie_id
         │ similarity_score     │   │
         │ common_users_count   │   │
         └──────────────────────┘───┘
```

---

## 4. 관계 요약

| 관계                  | 유형        | 설명                          |
|-----------------------|-------------|-------------------------------|
| users → ratings       | 1:N         | 한 사용자는 여러 평점 보유     |
| movies → ratings      | 1:N         | 한 영화는 여러 평점 보유       |
| movies ↔ genres       | N:M         | 영화-장르 다대다 (movie_genres를 통해) |
| movies ↔ movies       | N:M         | 영화 간 유사도 (movie_similarities를 통해) |

---

## 5. 데이터 규모 예상 (MovieLens 1M 기준)

| 테이블              | 예상 레코드 수 |
|---------------------|----------------|
| users               | ~6,040         |
| movies              | ~3,900         |
| genres              | 18             |
| movie_genres        | ~6,000         |
| ratings             | ~1,000,209     |
| ratings_train       | ~800,000       |
| ratings_test        | ~200,000       |
| movie_similarities  | ~100,000 ~ 500,000 (계산량에 따라 제한) |

---

## 6. 성능 최적화 전략

### 6.1 인덱스 전략
- 모든 FK에 인덱스 생성
- 자주 조회되는 컬럼 (timestamp, rating) 인덱스
- 복합 인덱스: (user_id, movie_id), (movie_id, rating)

### 6.2 파티셔닝 (선택적)
- ratings 테이블을 연도별 파티셔닝 (데이터 규모 증가 시)

### 6.3 Materialized View
- 인기 영화 Top 100 (매일 갱신)
- 장르별 인기 영화 (매일 갱신)

---

## 7. 마이그레이션 전략

### Phase 1
- users, movies, genres, movie_genres, ratings 테이블 생성
- ratings_train, ratings_test 분리

### Phase 2
- movie_similarities 테이블 추가
- Materialized View 추가
- movies 테이블에 description, poster_url 컬럼 추가 (TMDB API 연동 시)

---

## 8. 참고 사항

### 8.1 MovieLens 1M 데이터셋 원본 형식

**users.dat**: UserID::Gender::Age::Occupation::Zip-code
- 예: 1::F::1::10::48067

**movies.dat**: MovieID::Title::Genres
- 예: 1::Toy Story (1995)::Animation|Children's|Comedy

**ratings.dat**: UserID::MovieID::Rating::Timestamp
- 예: 1::1193::5::978300760

### 8.2 연령대 코드 (age)
- 1: "Under 18"
- 18: "18-24"
- 25: "25-34"
- 35: "35-44"
- 45: "45-49"
- 50: "50-55"
- 56: "56+"

### 8.3 직업 코드 (occupation)
- 0: "other"
- 1: "academic/educator"
- 2: "artist"
- 3: "clerical/admin"
- 4: "college/grad student"
- 5: "customer service"
- 6: "doctor/health care"
- 7: "executive/managerial"
- 8: "farmer"
- 9: "homemaker"
- 10: "K-12 student"
- 11: "lawyer"
- 12: "programmer"
- 13: "retired"
- 14: "sales/marketing"
- 15: "scientist"
- 16: "self-employed"
- 17: "technician/engineer"
- 18: "tradesman/craftsman"
- 19: "unemployed"
- 20: "writer"

---

문서 끝.

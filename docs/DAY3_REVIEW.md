# Day 3 코드 리뷰 및 수정 사항

날짜: 2025-12-05

## 구현 완료

### 1. 데이터 다운로드 및 변환
- MovieLens 1M 데이터셋 다운로드 성공
- DAT 파일 → CSV 변환 완료
  - users: 6,040개
  - movies: 3,883개
  - genres: 18개
  - movie_genres: 6,408개
  - ratings: 1,000,209개

### 2. 전처리
- `release_year` 추출: 정규식 `r'\((\d{4})\)$'` 사용
- 장르 파싱: `|` 구분자로 분리 후 genres, movie_genres 테이블 생성
- 평점 검증: 0.5~5.0 범위 확인

### 3. 데이터베이스 구축
- PostgreSQL 18.1 설치
- movielens_db 데이터베이스 생성
- 8개 테이블 생성 (users, movies, genres, movie_genres, ratings, ratings_train, ratings_test, movie_similarities)

## 발견한 문제 및 해결

### 문제 1: psycopg2 한글 경로 인코딩 에러
**증상:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb8 in position 63
```

**원인:** psycopg2가 한글 경로(`C:\claude\영화추천프로그램`)를 처리하지 못함

**해결:**
```python
# 수정 전
conn = psycopg2.connect(host=..., port=..., database=..., user=..., password=...)

# 수정 후
conn_string = f"host={host} port={port} dbname={database} user={user} password={password}"
conn = psycopg2.connect(conn_string)
```

### 문제 2: schema.sql 한글 COMMENT 인코딩 에러
**증상:**
```
psql: 오류: 0xec 0x82 바이트로 조합된 문자(인코딩: "UHC")와 대응되는 문자 코드가 "UTF8" 인코딩에는 없습니다
```

**원인:** schema.sql의 한글 COMMENT가 UHC로 읽힘

**해결:** 한글 주석을 모두 제거한 `schema_simple.sql` 생성

### 문제 3: public 스키마 권한 부족
**증상:**
```
psql: 오류: public 스키마(schema) 접근 권한 없음
```

**해결:**
```sql
GRANT ALL ON SCHEMA public TO movielens_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO movielens_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO movielens_user;
```

### 문제 4: Python PATH 미설정
**증상:** PowerShell에서 `python` 명령어 인식 불가

**해결:** 전체 경로 사용
```powershell
C:\Users\kobin\AppData\Local\Programs\Python\Python313\python.exe
```

## 코드 개선 사항

### data_loader.py
1. **파일 존재 확인 추가:**
```python
required_files = ['users.dat', 'movies.dat', 'ratings.dat']
for file_name in required_files:
    file_path = ml_1m_dir / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")
```

2. **에러 처리 강화:**
```python
try:
    users_df = pd.read_csv(...)
except Exception as e:
    logger.error(f"Failed to read users.dat: {e}")
    raise
```

3. **청크 단위 로딩:**
```python
chunk_size = 10000
for i in range(0, len(ratings_df), chunk_size):
    chunk = ratings_df.iloc[i:i+chunk_size]
    chunk.to_sql('ratings', conn, if_exists='append', index=False)
```

### setup_db.py
1. **연결 문자열 방식으로 변경**
2. **스키마 파일 다중 인코딩 지원 (미사용)**

## 다음 단계 (Day 4)

- EDA (탐색적 데이터 분석)
- Jupyter Notebook으로 데이터 분석
- 통계 및 시각화

## 성공 기준 달성

✅ 3가지 CSV 변환 완료
✅ PostgreSQL DB 생성 및 데이터 로딩 완료
✅ 데이터 검증 완료 (레코드 수 확인)

---
작성자: Claude Code
검토 완료: 2025-12-05

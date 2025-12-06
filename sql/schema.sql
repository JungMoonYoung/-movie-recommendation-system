-- MovieLens Recommendation System Database Schema
-- PostgreSQL 12+
-- Version: 1.0
-- Date: 2025-12-04

-- Drop tables if exist (for clean setup)
DROP TABLE IF EXISTS movie_similarities CASCADE;
DROP TABLE IF EXISTS ratings_test CASCADE;
DROP TABLE IF EXISTS ratings_train CASCADE;
DROP TABLE IF EXISTS ratings CASCADE;
DROP TABLE IF EXISTS movie_genres CASCADE;
DROP TABLE IF EXISTS genres CASCADE;
DROP TABLE IF EXISTS movies CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- ============================================================================
-- 1. users (사용자 정보)
-- ============================================================================
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    gender CHAR(1) CHECK (gender IN ('M', 'F')),
    age INTEGER CHECK (age >= 1 AND age <= 56),
    occupation INTEGER CHECK (occupation >= 0 AND occupation <= 20),
    zip_code VARCHAR(10)
);

COMMENT ON TABLE users IS '사용자 기본 정보 및 demographic 데이터';
COMMENT ON COLUMN users.user_id IS '사용자 ID';
COMMENT ON COLUMN users.gender IS '성별 (M/F)';
COMMENT ON COLUMN users.age IS '연령대 코드 (1: Under 18, 18: 18-24, 25: 25-34, ...)';
COMMENT ON COLUMN users.occupation IS '직업 코드 (0: other, 1: academic/educator, ...)';
COMMENT ON COLUMN users.zip_code IS '우편번호';

-- ============================================================================
-- 2. movies (영화 정보)
-- ============================================================================
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    release_year INTEGER CHECK (release_year >= 1900 AND release_year <= 2100)
);

CREATE INDEX idx_movies_release_year ON movies(release_year);
CREATE INDEX idx_movies_title ON movies(title);

COMMENT ON TABLE movies IS '영화 기본 메타데이터';
COMMENT ON COLUMN movies.movie_id IS '영화 ID';
COMMENT ON COLUMN movies.title IS '영화 제목 (연도 제거됨)';
COMMENT ON COLUMN movies.release_year IS '개봉 연도';

-- ============================================================================
-- 3. genres (장르 정보)
-- ============================================================================
CREATE TABLE genres (
    genre_id SERIAL PRIMARY KEY,
    genre_name VARCHAR(50) NOT NULL UNIQUE
);

COMMENT ON TABLE genres IS '영화 장르 마스터 데이터';
COMMENT ON COLUMN genres.genre_id IS '장르 ID (자동 증가)';
COMMENT ON COLUMN genres.genre_name IS '장르 이름';

-- ============================================================================
-- 4. movie_genres (영화-장르 다대다 관계)
-- ============================================================================
CREATE TABLE movie_genres (
    movie_id INTEGER NOT NULL,
    genre_id INTEGER NOT NULL,
    PRIMARY KEY (movie_id, genre_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id) ON DELETE CASCADE
);

CREATE INDEX idx_movie_genres_genre_id ON movie_genres(genre_id);

COMMENT ON TABLE movie_genres IS '영화와 장르의 다대다 관계';
COMMENT ON COLUMN movie_genres.movie_id IS '영화 ID';
COMMENT ON COLUMN movie_genres.genre_id IS '장르 ID';

-- ============================================================================
-- 5. ratings (평점 정보)
-- ============================================================================
CREATE TABLE ratings (
    rating_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating DECIMAL(2,1) NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
    timestamp BIGINT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    UNIQUE (user_id, movie_id)
);

CREATE INDEX idx_ratings_user_id ON ratings(user_id);
CREATE INDEX idx_ratings_movie_id ON ratings(movie_id);
CREATE INDEX idx_ratings_timestamp ON ratings(timestamp);
CREATE INDEX idx_ratings_rating ON ratings(rating);

COMMENT ON TABLE ratings IS '사용자가 영화에 부여한 평점';
COMMENT ON COLUMN ratings.rating_id IS '평점 ID (자동 증가)';
COMMENT ON COLUMN ratings.user_id IS '사용자 ID';
COMMENT ON COLUMN ratings.movie_id IS '영화 ID';
COMMENT ON COLUMN ratings.rating IS '평점 (0.5~5.0, 0.5 단위)';
COMMENT ON COLUMN ratings.timestamp IS '평점 등록 시간 (Unix timestamp)';

-- ============================================================================
-- 6. ratings_train (학습 데이터)
-- ============================================================================
CREATE TABLE ratings_train (
    rating_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating DECIMAL(2,1) NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
    timestamp BIGINT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE
);

CREATE INDEX idx_ratings_train_user_id ON ratings_train(user_id);
CREATE INDEX idx_ratings_train_movie_id ON ratings_train(movie_id);

COMMENT ON TABLE ratings_train IS 'Train 데이터 (과거 80%)';

-- ============================================================================
-- 7. ratings_test (테스트 데이터)
-- ============================================================================
CREATE TABLE ratings_test (
    rating_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating DECIMAL(2,1) NOT NULL CHECK (rating >= 0.5 AND rating <= 5.0),
    timestamp BIGINT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE
);

CREATE INDEX idx_ratings_test_user_id ON ratings_test(user_id);
CREATE INDEX idx_ratings_test_movie_id ON ratings_test(movie_id);

COMMENT ON TABLE ratings_test IS 'Test 데이터 (최근 20%)';

-- ============================================================================
-- 8. movie_similarities (영화 유사도 캐시)
-- ============================================================================
CREATE TABLE movie_similarities (
    movie_id_1 INTEGER NOT NULL,
    movie_id_2 INTEGER NOT NULL,
    similarity_score DECIMAL(5,4) NOT NULL CHECK (similarity_score >= -1.0 AND similarity_score <= 1.0),
    common_users_count INTEGER NOT NULL CHECK (common_users_count >= 20),
    PRIMARY KEY (movie_id_1, movie_id_2),
    FOREIGN KEY (movie_id_1) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id_2) REFERENCES movies(movie_id) ON DELETE CASCADE,
    CHECK (movie_id_1 < movie_id_2)
);

CREATE INDEX idx_movie_similarities_movie_id_1 ON movie_similarities(movie_id_1);
CREATE INDEX idx_movie_similarities_score ON movie_similarities(similarity_score DESC);

COMMENT ON TABLE movie_similarities IS 'Item-based CF를 위한 영화 간 유사도';
COMMENT ON COLUMN movie_similarities.movie_id_1 IS '영화 ID 1 (작은 값)';
COMMENT ON COLUMN movie_similarities.movie_id_2 IS '영화 ID 2 (큰 값)';
COMMENT ON COLUMN movie_similarities.similarity_score IS '유사도 (Pearson correlation, -1.0 ~ 1.0)';
COMMENT ON COLUMN movie_similarities.common_users_count IS '공통 평가 사용자 수';

-- ============================================================================
-- Materialized Views (Phase 2)
-- ============================================================================

-- NOTE: Materialized View는 데이터 로딩 후 별도로 생성하는 것을 권장
-- 아래 명령어는 sql/views/mv_popular_movies.sql로 분리

-- 인기 영화 Top 100 (전체 사용자 기준)
-- 데이터 로딩 후 실행:
-- CREATE MATERIALIZED VIEW mv_popular_movies AS ...

-- Phase 1에서는 뷰 생성 생략 (데이터 없을 때 생성하면 빈 뷰 생성됨)

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- 유효한 평점인지 확인하는 함수
CREATE OR REPLACE FUNCTION is_valid_rating(r DECIMAL)
RETURNS BOOLEAN AS $$
BEGIN
    -- 0.5~5.0 범위이고, (rating - 0.5) / 0.5가 정수인지 확인
    RETURN r >= 0.5 AND r <= 5.0 AND MOD(((r - 0.5) * 10)::INTEGER, 5) = 0;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION is_valid_rating IS '평점이 0.5 단위로 0.5~5.0 범위인지 확인';

-- ============================================================================
-- Grants (권한 설정)
-- ============================================================================

-- movielens_user가 존재하는 경우에만 권한 부여
-- 사용자 생성은 별도로 수행해야 함 (CREATE USER movielens_user WITH PASSWORD 'your_password';)

DO $$
BEGIN
    IF EXISTS (SELECT FROM pg_user WHERE usename = 'movielens_user') THEN
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO movielens_user;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO movielens_user;
        GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO movielens_user;
        RAISE NOTICE 'Granted privileges to movielens_user';
    ELSE
        RAISE NOTICE 'User movielens_user does not exist. Skipping grants.';
    END IF;
END
$$;

-- ============================================================================
-- Statistics Update
-- ============================================================================

-- 통계 정보 업데이트 (데이터 로딩 후 실행)
-- ANALYZE users;
-- ANALYZE movies;
-- ANALYZE genres;
-- ANALYZE movie_genres;
-- ANALYZE ratings;
-- ANALYZE ratings_train;
-- ANALYZE ratings_test;

-- ============================================================================
-- 스키마 생성 완료
-- ============================================================================

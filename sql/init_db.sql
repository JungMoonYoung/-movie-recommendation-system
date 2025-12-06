-- Database Initialization Script
-- 데이터베이스 및 사용자 생성
-- PostgreSQL superuser (postgres)로 실행

-- 1. 기존 데이터베이스 삭제 (주의: 데이터 손실!)
-- DROP DATABASE IF EXISTS movielens_db;

-- 2. 데이터베이스 생성
CREATE DATABASE movielens_db
    WITH
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    TEMPLATE = template0;

COMMENT ON DATABASE movielens_db IS 'MovieLens 추천 시스템 데이터베이스';

-- 3. 사용자 생성 (이미 존재하면 건너뜀)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'movielens_user') THEN
        CREATE USER movielens_user WITH PASSWORD 'movielens_pass';
        RAISE NOTICE 'User movielens_user created';
    ELSE
        RAISE NOTICE 'User movielens_user already exists';
    END IF;
END
$$;

-- 4. 권한 부여
GRANT ALL PRIVILEGES ON DATABASE movielens_db TO movielens_user;

-- 5. movielens_db에 연결 후 스키마 생성
-- \c movielens_db
-- \i sql/schema.sql

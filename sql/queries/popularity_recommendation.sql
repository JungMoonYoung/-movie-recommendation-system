-- Popularity-based Recommendation Query
-- 인기 기반 추천: 전체 사용자의 평점을 기준으로 인기 있는 영화 추천

-- 기본 인기 영화 추천 (최소 평점 수 필터링)
-- Parameters:
--   - min_ratings: 최소 평점 개수 (기본값: 30)
--   - top_n: 추천할 영화 개수 (기본값: 10)

WITH popular_movies AS (
    SELECT
        m.movie_id,
        m.title,
        m.release_year,
        COUNT(r.rating_id) as rating_count,
        AVG(r.rating) as avg_rating,
        -- 가중 평점 계산 (Bayesian Average)
        -- WR = (v / (v + m)) * R + (m / (v + m)) * C
        -- v: 영화의 평점 개수, m: 최소 평점 기준, R: 평균 평점, C: 전체 평균
        (COUNT(r.rating_id)::FLOAT / (COUNT(r.rating_id) + 30)) * AVG(r.rating) +
        (30.0 / (COUNT(r.rating_id) + 30)) * (SELECT AVG(rating) FROM ratings_train)
        as weighted_rating
    FROM movies m
    INNER JOIN ratings_train r ON m.movie_id = r.movie_id
    GROUP BY m.movie_id, m.title, m.release_year
    HAVING COUNT(r.rating_id) >= 30  -- 최소 평점 수 필터
)
SELECT
    movie_id,
    title,
    release_year,
    rating_count,
    ROUND(avg_rating::NUMERIC, 2) as avg_rating,
    ROUND(weighted_rating::NUMERIC, 2) as weighted_rating
FROM popular_movies
ORDER BY weighted_rating DESC, rating_count DESC
LIMIT 10;

-- 사용자 맞춤 인기 영화 추천 (이미 본 영화 제외)
-- Parameters:
--   - user_id: 사용자 ID
--   - min_ratings: 최소 평점 개수
--   - top_n: 추천할 영화 개수

-- Example usage:
-- WITH user_watched AS (
--     SELECT movie_id
--     FROM ratings_train
--     WHERE user_id = 1
-- ),
-- popular_movies AS (
--     SELECT
--         m.movie_id,
--         m.title,
--         m.release_year,
--         COUNT(r.rating_id) as rating_count,
--         AVG(r.rating) as avg_rating,
--         (COUNT(r.rating_id)::FLOAT / (COUNT(r.rating_id) + 30)) * AVG(r.rating) +
--         (30.0 / (COUNT(r.rating_id) + 30)) * (SELECT AVG(rating) FROM ratings_train)
--         as weighted_rating
--     FROM movies m
--     INNER JOIN ratings_train r ON m.movie_id = r.movie_id
--     WHERE m.movie_id NOT IN (SELECT movie_id FROM user_watched)
--     GROUP BY m.movie_id, m.title, m.release_year
--     HAVING COUNT(r.rating_id) >= 30
-- )
-- SELECT
--     movie_id,
--     title,
--     release_year,
--     rating_count,
--     ROUND(avg_rating::NUMERIC, 2) as avg_rating,
--     ROUND(weighted_rating::NUMERIC, 2) as weighted_rating
-- FROM popular_movies
-- ORDER BY weighted_rating DESC, rating_count DESC
-- LIMIT 10;

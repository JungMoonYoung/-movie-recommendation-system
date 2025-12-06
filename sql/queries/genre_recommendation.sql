-- Genre-based Recommendation Query
-- 장르 기반 추천: 사용자의 선호 장르를 분석하여 해당 장르의 인기 영화 추천

-- Step 1: 사용자 선호 장르 분석
-- Parameters:
--   - user_id: 사용자 ID
--   - min_ratings_per_genre: 장르당 최소 평점 개수 (기본값: 3)
--   - top_genres: 상위 N개 장르 선택 (기본값: 3)

-- Example: Get user's top preferred genres
-- WITH user_genre_preference AS (
--     SELECT
--         r.user_id,
--         g.genre_name,
--         COUNT(r.rating_id) as rating_count,
--         AVG(r.rating) as avg_rating,
--         -- Preference score: rating_count * avg_rating
--         COUNT(r.rating_id) * AVG(r.rating) as preference_score
--     FROM ratings_train r
--     INNER JOIN movie_genres mg ON r.movie_id = mg.movie_id
--     INNER JOIN genres g ON mg.genre_id = g.genre_id
--     WHERE r.user_id = 1
--     GROUP BY r.user_id, g.genre_name
--     HAVING COUNT(r.rating_id) >= 3
-- )
-- SELECT
--     genre_name,
--     rating_count,
--     ROUND(avg_rating::NUMERIC, 2) as avg_rating,
--     ROUND(preference_score::NUMERIC, 2) as preference_score
-- FROM user_genre_preference
-- ORDER BY preference_score DESC
-- LIMIT 3;

-- Step 2: 사용자 선호 장르의 인기 영화 추천
-- Parameters:
--   - user_id: 사용자 ID
--   - min_movie_ratings: 영화당 최소 평점 개수 (기본값: 20)
--   - top_n: 추천할 영화 개수 (기본값: 10)

-- Complete recommendation query
WITH user_genre_preference AS (
    -- 사용자 선호 장르 파악
    SELECT
        r.user_id,
        g.genre_id,
        g.genre_name,
        COUNT(r.rating_id) as rating_count,
        AVG(r.rating) as avg_rating,
        COUNT(r.rating_id) * AVG(r.rating) as preference_score
    FROM ratings_train r
    INNER JOIN movie_genres mg ON r.movie_id = mg.movie_id
    INNER JOIN genres g ON mg.genre_id = g.genre_id
    WHERE r.user_id = 1  -- Replace with parameter
    GROUP BY r.user_id, g.genre_id, g.genre_name
    HAVING COUNT(r.rating_id) >= 3
    ORDER BY preference_score DESC
    LIMIT 3  -- Top 3 preferred genres
),
user_watched AS (
    -- 사용자가 이미 본 영화
    SELECT movie_id
    FROM ratings_train
    WHERE user_id = 1  -- Replace with parameter
),
genre_movies AS (
    -- 선호 장르의 영화들
    SELECT DISTINCT
        m.movie_id,
        m.title,
        m.release_year,
        ugp.genre_name,
        ugp.preference_score as genre_preference
    FROM movies m
    INNER JOIN movie_genres mg ON m.movie_id = mg.movie_id
    INNER JOIN user_genre_preference ugp ON mg.genre_id = ugp.genre_id
    WHERE m.movie_id NOT IN (SELECT movie_id FROM user_watched)
),
movie_stats AS (
    -- 영화별 통계
    SELECT
        gm.movie_id,
        gm.title,
        gm.release_year,
        gm.genre_name,
        gm.genre_preference,
        COUNT(r.rating_id) as rating_count,
        AVG(r.rating) as avg_rating,
        -- Weighted rating (Bayesian Average)
        (COUNT(r.rating_id)::FLOAT / (COUNT(r.rating_id) + 20)) * AVG(r.rating) +
        (20::FLOAT / (COUNT(r.rating_id) + 20)) * (SELECT AVG(rating) FROM ratings_train)
        as weighted_rating,
        -- Combined score: weighted_rating * genre_preference (normalized)
        ((COUNT(r.rating_id)::FLOAT / (COUNT(r.rating_id) + 20)) * AVG(r.rating) +
         (20::FLOAT / (COUNT(r.rating_id) + 20)) * (SELECT AVG(rating) FROM ratings_train))
        * (gm.genre_preference / 100.0) as combined_score
    FROM genre_movies gm
    INNER JOIN ratings_train r ON gm.movie_id = r.movie_id
    GROUP BY gm.movie_id, gm.title, gm.release_year, gm.genre_name, gm.genre_preference
    HAVING COUNT(r.rating_id) >= 20  -- Minimum rating count
)
SELECT
    movie_id,
    title,
    release_year,
    genre_name as preferred_genre,
    rating_count,
    ROUND(avg_rating::NUMERIC, 2) as avg_rating,
    ROUND(weighted_rating::NUMERIC, 2) as weighted_rating,
    ROUND(combined_score::NUMERIC, 2) as recommendation_score
FROM movie_stats
ORDER BY combined_score DESC, weighted_rating DESC
LIMIT 10;

-- Alternative: Multi-genre support (if movie belongs to multiple preferred genres)
-- This version gives bonus to movies that match multiple preferred genres

-- WITH user_genre_preference AS (
--     SELECT
--         r.user_id,
--         g.genre_id,
--         g.genre_name,
--         COUNT(r.rating_id) as rating_count,
--         AVG(r.rating) as avg_rating,
--         COUNT(r.rating_id) * AVG(r.rating) as preference_score,
--         ROW_NUMBER() OVER (PARTITION BY r.user_id ORDER BY COUNT(r.rating_id) * AVG(r.rating) DESC) as genre_rank
--     FROM ratings_train r
--     INNER JOIN movie_genres mg ON r.movie_id = mg.movie_id
--     INNER JOIN genres g ON mg.genre_id = g.genre_id
--     WHERE r.user_id = 1
--     GROUP BY r.user_id, g.genre_id, g.genre_name
--     HAVING COUNT(r.rating_id) >= 3
-- ),
-- user_watched AS (
--     SELECT movie_id
--     FROM ratings_train
--     WHERE user_id = 1
-- ),
-- movie_genre_match AS (
--     SELECT
--         m.movie_id,
--         m.title,
--         m.release_year,
--         COUNT(DISTINCT ugp.genre_id) as matched_genres,
--         SUM(ugp.preference_score) as total_preference,
--         STRING_AGG(ugp.genre_name, ', ' ORDER BY ugp.preference_score DESC) as matched_genre_names
--     FROM movies m
--     INNER JOIN movie_genres mg ON m.movie_id = mg.movie_id
--     INNER JOIN user_genre_preference ugp ON mg.genre_id = ugp.genre_id AND ugp.genre_rank <= 3
--     WHERE m.movie_id NOT IN (SELECT movie_id FROM user_watched)
--     GROUP BY m.movie_id, m.title, m.release_year
-- ),
-- movie_stats AS (
--     SELECT
--         mgm.movie_id,
--         mgm.title,
--         mgm.release_year,
--         mgm.matched_genres,
--         mgm.matched_genre_names,
--         COUNT(r.rating_id) as rating_count,
--         AVG(r.rating) as avg_rating,
--         (COUNT(r.rating_id)::FLOAT / (COUNT(r.rating_id) + 20)) * AVG(r.rating) +
--         (20::FLOAT / (COUNT(r.rating_id) + 20)) * (SELECT AVG(rating) FROM ratings_train)
--         as weighted_rating,
--         -- Genre boost: multiply by matched_genres for multi-genre bonus
--         ((COUNT(r.rating_id)::FLOAT / (COUNT(r.rating_id) + 20)) * AVG(r.rating) +
--          (20::FLOAT / (COUNT(r.rating_id) + 20)) * (SELECT AVG(rating) FROM ratings_train))
--         * mgm.matched_genres * (mgm.total_preference / 100.0) as recommendation_score
--     FROM movie_genre_match mgm
--     INNER JOIN ratings_train r ON mgm.movie_id = r.movie_id
--     GROUP BY mgm.movie_id, mgm.title, mgm.release_year, mgm.matched_genres, mgm.matched_genre_names, mgm.total_preference
--     HAVING COUNT(r.rating_id) >= 20
-- )
-- SELECT
--     movie_id,
--     title,
--     release_year,
--     matched_genre_names as genres,
--     matched_genres as genre_match_count,
--     rating_count,
--     ROUND(avg_rating::NUMERIC, 2) as avg_rating,
--     ROUND(weighted_rating::NUMERIC, 2) as weighted_rating,
--     ROUND(recommendation_score::NUMERIC, 2) as recommendation_score
-- FROM movie_stats
-- ORDER BY recommendation_score DESC, matched_genres DESC, weighted_rating DESC
-- LIMIT 10;

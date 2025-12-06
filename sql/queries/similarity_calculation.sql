-- Movie Similarity Calculation using Cosine Similarity
-- 영화 간 유사도 계산 (코사인 유사도 사용)

-- Cosine Similarity 공식:
-- similarity = (A · B) / (||A|| × ||B||)
-- where A, B are rating vectors for two movies

-- Step 1: 두 영화에 대해 공통 사용자의 평점 벡터 추출
-- Parameters:
--   - movie_id_1: 첫 번째 영화 ID
--   - movie_id_2: 두 번째 영화 ID
--   - min_common_users: 최소 공통 사용자 수 (기본값: 20)

-- Example: Calculate similarity between two specific movies
-- WITH common_ratings AS (
--     SELECT
--         r1.user_id,
--         r1.rating as rating_1,
--         r2.rating as rating_2
--     FROM ratings_train r1
--     INNER JOIN ratings_train r2
--         ON r1.user_id = r2.user_id
--     WHERE r1.movie_id = 1  -- movie_id_1
--       AND r2.movie_id = 2  -- movie_id_2
-- ),
-- similarity_calc AS (
--     SELECT
--         COUNT(*) as common_users,
--         -- Dot product: sum(r1 * r2)
--         SUM(rating_1 * rating_2) as dot_product,
--         -- Magnitude of vector 1: sqrt(sum(r1^2))
--         SQRT(SUM(rating_1 * rating_1)) as magnitude_1,
--         -- Magnitude of vector 2: sqrt(sum(r2^2))
--         SQRT(SUM(rating_2 * rating_2)) as magnitude_2
--     FROM common_ratings
-- )
-- SELECT
--     common_users,
--     CASE
--         WHEN magnitude_1 > 0 AND magnitude_2 > 0 THEN
--             dot_product / (magnitude_1 * magnitude_2)
--         ELSE 0
--     END as cosine_similarity
-- FROM similarity_calc
-- WHERE common_users >= 20;

-- Step 2: 모든 영화 쌍에 대한 유사도 계산 (배치 처리용)
-- This query calculates similarities for all movie pairs with sufficient common users
-- WARNING: This is computationally expensive! Use with LIMIT or WHERE clause.

-- Optimized version: Calculate for specific movie
WITH target_movie_users AS (
    -- Users who rated the target movie
    SELECT
        user_id,
        rating
    FROM ratings_train
    WHERE movie_id = 1  -- Replace with target movie_id
),
candidate_movies AS (
    -- Movies rated by users who also rated target movie
    SELECT DISTINCT r.movie_id
    FROM ratings_train r
    WHERE r.user_id IN (SELECT user_id FROM target_movie_users)
      AND r.movie_id != 1  -- Exclude target movie
),
similarity_scores AS (
    SELECT
        cm.movie_id as other_movie_id,
        COUNT(r1.user_id) as common_users,
        -- Cosine similarity calculation
        SUM(r1.rating * r2.rating) as dot_product,
        SQRT(SUM(r1.rating * r1.rating)) as magnitude_1,
        SQRT(SUM(r2.rating * r2.rating)) as magnitude_2
    FROM candidate_movies cm
    INNER JOIN ratings_train r1
        ON r1.user_id IN (SELECT user_id FROM target_movie_users)
        AND r1.movie_id = 1  -- target movie
    INNER JOIN ratings_train r2
        ON r2.user_id = r1.user_id
        AND r2.movie_id = cm.movie_id
    GROUP BY cm.movie_id
    HAVING COUNT(r1.user_id) >= 20  -- Minimum common users
)
SELECT
    1 as movie_id_1,
    other_movie_id as movie_id_2,
    common_users,
    CASE
        WHEN magnitude_1 > 0 AND magnitude_2 > 0 THEN
            ROUND((dot_product / (magnitude_1 * magnitude_2))::NUMERIC, 4)
        ELSE 0
    END as similarity_score
FROM similarity_scores
WHERE magnitude_1 > 0 AND magnitude_2 > 0
ORDER BY similarity_score DESC
LIMIT 50;

-- Step 3: Batch calculation for popular movies
-- Calculate similarities for top N most-rated movies only (to reduce computation)

-- WITH popular_movies AS (
--     SELECT
--         movie_id,
--         COUNT(*) as rating_count
--     FROM ratings_train
--     GROUP BY movie_id
--     HAVING COUNT(*) >= 100  -- Only movies with 100+ ratings
--     ORDER BY COUNT(*) DESC
--     LIMIT 500  -- Top 500 popular movies
-- ),
-- movie_pairs AS (
--     -- Generate all pairs (movie_id_1 < movie_id_2 to avoid duplicates)
--     SELECT
--         m1.movie_id as movie_id_1,
--         m2.movie_id as movie_id_2
--     FROM popular_movies m1
--     CROSS JOIN popular_movies m2
--     WHERE m1.movie_id < m2.movie_id
-- ),
-- pair_similarities AS (
--     SELECT
--         mp.movie_id_1,
--         mp.movie_id_2,
--         COUNT(r1.user_id) as common_users,
--         SUM(r1.rating * r2.rating) as dot_product,
--         SQRT(SUM(r1.rating * r1.rating)) as magnitude_1,
--         SQRT(SUM(r2.rating * r2.rating)) as magnitude_2
--     FROM movie_pairs mp
--     INNER JOIN ratings_train r1 ON r1.movie_id = mp.movie_id_1
--     INNER JOIN ratings_train r2
--         ON r2.movie_id = mp.movie_id_2
--         AND r2.user_id = r1.user_id
--     GROUP BY mp.movie_id_1, mp.movie_id_2
--     HAVING COUNT(r1.user_id) >= 20
-- )
-- SELECT
--     movie_id_1,
--     movie_id_2,
--     common_users,
--     CASE
--         WHEN magnitude_1 > 0 AND magnitude_2 > 0 THEN
--             ROUND((dot_product / (magnitude_1 * magnitude_2))::NUMERIC, 4)
--         ELSE 0
--     END as similarity_score
-- FROM pair_similarities
-- WHERE magnitude_1 > 0 AND magnitude_2 > 0
-- ORDER BY similarity_score DESC;

-- Note: Full batch calculation should be done in Python with progress tracking
-- Use this SQL as a template for Python implementation

-- Materialized View: 인기 영화 Top 100
-- 데이터 로딩 완료 후 실행

-- Drop if exists
DROP MATERIALIZED VIEW IF EXISTS mv_popular_movies CASCADE;

-- Create materialized view
CREATE MATERIALIZED VIEW mv_popular_movies AS
SELECT
    m.movie_id,
    m.title,
    m.release_year,
    COUNT(r.rating) AS rating_count,
    AVG(r.rating) AS avg_rating,
    ROUND(AVG(r.rating), 2) AS avg_rating_rounded
FROM movies m
JOIN ratings_train r ON m.movie_id = r.movie_id
GROUP BY m.movie_id, m.title, m.release_year
HAVING COUNT(r.rating) >= 30
ORDER BY avg_rating DESC, rating_count DESC
LIMIT 100;

-- Create index
CREATE INDEX idx_mv_popular_movies_movie_id ON mv_popular_movies(movie_id);

-- Comment
COMMENT ON MATERIALIZED VIEW mv_popular_movies IS '인기 영화 Top 100 (배치 갱신용)';

-- Refresh view (필요 시 실행)
-- REFRESH MATERIALIZED VIEW mv_popular_movies;

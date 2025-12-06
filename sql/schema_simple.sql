-- MovieLens Recommendation System Database Schema
-- PostgreSQL 18+

DROP TABLE IF EXISTS movie_similarities CASCADE;
DROP TABLE IF EXISTS ratings_test CASCADE;
DROP TABLE IF EXISTS ratings_train CASCADE;
DROP TABLE IF EXISTS ratings CASCADE;
DROP TABLE IF EXISTS movie_genres CASCADE;
DROP TABLE IF EXISTS genres CASCADE;
DROP TABLE IF EXISTS movies CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- 1. users
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    gender CHAR(1) CHECK (gender IN ('M', 'F')),
    age INTEGER CHECK (age >= 1 AND age <= 56),
    occupation INTEGER CHECK (occupation >= 0 AND occupation <= 20),
    zip_code VARCHAR(10)
);

-- 2. movies
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    release_year INTEGER CHECK (release_year >= 1900 AND release_year <= 2100)
);

CREATE INDEX idx_movies_release_year ON movies(release_year);
CREATE INDEX idx_movies_title ON movies(title);

-- 3. genres
CREATE TABLE genres (
    genre_id SERIAL PRIMARY KEY,
    genre_name VARCHAR(50) NOT NULL UNIQUE
);

-- 4. movie_genres
CREATE TABLE movie_genres (
    movie_id INTEGER NOT NULL,
    genre_id INTEGER NOT NULL,
    PRIMARY KEY (movie_id, genre_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(genre_id) ON DELETE CASCADE
);

CREATE INDEX idx_movie_genres_genre_id ON movie_genres(genre_id);

-- 5. ratings
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

-- 6. ratings_train
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

-- 7. ratings_test
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

-- 8. movie_similarities
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

"""
Data loading and preprocessing for MovieLens 1M dataset
"""
import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from typing import Tuple

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DB_CONFIG,
    MIN_RATING,
    MAX_RATING
)
from src.db_connection import get_db_connection, get_sqlalchemy_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_dat_to_csv():
    """
    Convert MovieLens .dat files to CSV format
    """
    logger.info("Converting .dat files to CSV...")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ml_1m_dir = RAW_DATA_DIR / 'ml-1m'

    # Verify required files exist
    required_files = ['users.dat', 'movies.dat', 'ratings.dat']
    for file_name in required_files:
        file_path = ml_1m_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    try:
        # 1. Users
        users_df = pd.read_csv(
            ml_1m_dir / 'users.dat',
            sep='::',
            engine='python',
            names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
            encoding='latin-1'
        )
    except Exception as e:
        logger.error(f"Failed to read users.dat: {e}")
        raise
    users_csv = PROCESSED_DATA_DIR / 'users.csv'
    users_df.to_csv(users_csv, index=False)
    logger.info(f"✓ Converted users.dat → {users_csv} ({len(users_df)} records)")

    try:
        # 2. Movies
        movies_df = pd.read_csv(
            ml_1m_dir / 'movies.dat',
            sep='::',
            engine='python',
            names=['movie_id', 'title', 'genres'],
            encoding='latin-1'
        )
    except Exception as e:
        logger.error(f"Failed to read movies.dat: {e}")
        raise

    # Extract release year from title using regex
    def extract_year(title):
        match = re.search(r'\((\d{4})\)$', title.strip())
        if match:
            return int(match.group(1))
        return None

    def remove_year_from_title(title):
        # Remove (YYYY) from the end of title
        return re.sub(r'\s*\(\d{4}\)$', '', title.strip())

    movies_df['release_year'] = movies_df['title'].apply(extract_year)
    movies_df['title'] = movies_df['title'].apply(remove_year_from_title)

    # Save movies (without genres column for now)
    movies_csv = PROCESSED_DATA_DIR / 'movies.csv'
    movies_df[['movie_id', 'title', 'release_year']].to_csv(movies_csv, index=False)
    logger.info(f"✓ Converted movies.dat → {movies_csv} ({len(movies_df)} records)")

    # Parse genres and create genres + movie_genres tables
    all_genres = set()
    movie_genre_list = []

    for idx, row in movies_df.iterrows():
        movie_id = row['movie_id']
        genres_str = row['genres']
        genres = genres_str.split('|')

        for genre in genres:
            all_genres.add(genre)
            movie_genre_list.append({'movie_id': movie_id, 'genre_name': genre})

    # Create genres DataFrame
    genres_df = pd.DataFrame(list(all_genres), columns=['genre_name'])
    genres_df = genres_df.sort_values('genre_name').reset_index(drop=True)
    genres_df['genre_id'] = genres_df.index + 1
    genres_df = genres_df[['genre_id', 'genre_name']]

    genres_csv = PROCESSED_DATA_DIR / 'genres.csv'
    genres_df.to_csv(genres_csv, index=False)
    logger.info(f"✓ Created genres.csv ({len(genres_df)} genres)")

    # Create movie_genres DataFrame with genre_id
    movie_genres_df = pd.DataFrame(movie_genre_list)
    movie_genres_df = movie_genres_df.merge(genres_df, on='genre_name')
    movie_genres_df = movie_genres_df[['movie_id', 'genre_id']]

    movie_genres_csv = PROCESSED_DATA_DIR / 'movie_genres.csv'
    movie_genres_df.to_csv(movie_genres_csv, index=False)
    logger.info(f"✓ Created movie_genres.csv ({len(movie_genres_df)} records)")

    try:
        # 3. Ratings
        ratings_df = pd.read_csv(
            ml_1m_dir / 'ratings.dat',
            sep='::',
            engine='python',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
    except Exception as e:
        logger.error(f"Failed to read ratings.dat: {e}")
        raise

    # Validate ratings
    invalid_ratings = ratings_df[
        (ratings_df['rating'] < MIN_RATING) |
        (ratings_df['rating'] > MAX_RATING)
    ]
    if len(invalid_ratings) > 0:
        logger.warning(f"Found {len(invalid_ratings)} invalid ratings. Removing...")
        ratings_df = ratings_df[
            (ratings_df['rating'] >= MIN_RATING) &
            (ratings_df['rating'] <= MAX_RATING)
        ]

    ratings_csv = PROCESSED_DATA_DIR / 'ratings.csv'
    ratings_df.to_csv(ratings_csv, index=False)
    logger.info(f"✓ Converted ratings.dat → {ratings_csv} ({len(ratings_df)} records)")

    logger.info("CSV conversion completed!")
    return {
        'users': users_csv,
        'movies': movies_csv,
        'genres': genres_csv,
        'movie_genres': movie_genres_csv,
        'ratings': ratings_csv
    }


def load_users(engine):
    """
    Load users data to PostgreSQL
    """
    logger.info("Loading users...")

    users_csv = PROCESSED_DATA_DIR / 'users.csv'
    users_df = pd.read_csv(users_csv)

    with engine.connect() as conn:
        # Truncate table
        conn.execute(text("TRUNCATE TABLE users CASCADE"))
        conn.commit()

        # Insert data
        users_df.to_sql('users', conn, if_exists='append', index=False)
        conn.commit()

    logger.info(f"✓ Loaded {len(users_df)} users")
    return len(users_df)


def load_movies(engine):
    """
    Load movies data to PostgreSQL
    """
    logger.info("Loading movies...")

    movies_csv = PROCESSED_DATA_DIR / 'movies.csv'
    movies_df = pd.read_csv(movies_csv)

    with engine.connect() as conn:
        # Truncate table
        conn.execute(text("TRUNCATE TABLE movies CASCADE"))
        conn.commit()

        # Insert data
        movies_df.to_sql('movies', conn, if_exists='append', index=False)
        conn.commit()

    logger.info(f"✓ Loaded {len(movies_df)} movies")
    return len(movies_df)


def load_genres(engine):
    """
    Load genres data to PostgreSQL
    """
    logger.info("Loading genres...")

    genres_csv = PROCESSED_DATA_DIR / 'genres.csv'
    genres_df = pd.read_csv(genres_csv)

    with engine.connect() as conn:
        # Truncate table
        conn.execute(text("TRUNCATE TABLE genres CASCADE"))
        conn.commit()

        # Reset sequence
        conn.execute(text("ALTER SEQUENCE genres_genre_id_seq RESTART WITH 1"))
        conn.commit()

        # Insert data
        genres_df.to_sql('genres', conn, if_exists='append', index=False)
        conn.commit()

    logger.info(f"✓ Loaded {len(genres_df)} genres")
    return len(genres_df)


def load_movie_genres(engine):
    """
    Load movie_genres relationship to PostgreSQL
    """
    logger.info("Loading movie_genres...")

    movie_genres_csv = PROCESSED_DATA_DIR / 'movie_genres.csv'
    movie_genres_df = pd.read_csv(movie_genres_csv)

    with engine.connect() as conn:
        # Truncate table
        conn.execute(text("TRUNCATE TABLE movie_genres CASCADE"))
        conn.commit()

        # Insert data
        movie_genres_df.to_sql('movie_genres', conn, if_exists='append', index=False)
        conn.commit()

    logger.info(f"✓ Loaded {len(movie_genres_df)} movie-genre relationships")
    return len(movie_genres_df)


def load_ratings(engine):
    """
    Load ratings data to PostgreSQL
    """
    logger.info("Loading ratings...")

    ratings_csv = PROCESSED_DATA_DIR / 'ratings.csv'
    ratings_df = pd.read_csv(ratings_csv)

    with engine.connect() as conn:
        # Truncate table
        conn.execute(text("TRUNCATE TABLE ratings CASCADE"))
        conn.commit()

        # Reset sequence
        conn.execute(text("ALTER SEQUENCE ratings_rating_id_seq RESTART WITH 1"))
        conn.commit()

        # Insert data in chunks for better performance
        chunk_size = 10000
        for i in range(0, len(ratings_df), chunk_size):
            chunk = ratings_df.iloc[i:i+chunk_size]
            chunk.to_sql('ratings', conn, if_exists='append', index=False)
            logger.info(f"  Inserted {min(i+chunk_size, len(ratings_df))}/{len(ratings_df)} ratings")

        conn.commit()

    logger.info(f"✓ Loaded {len(ratings_df)} ratings")
    return len(ratings_df)


def load_all_data():
    """
    Complete data loading pipeline
    """
    logger.info("=" * 60)
    logger.info("Starting data loading pipeline...")
    logger.info("=" * 60)

    # Step 1: Convert DAT to CSV
    try:
        csv_files = convert_dat_to_csv()
    except Exception as e:
        logger.error(f"Failed to convert DAT files: {e}")
        raise

    # Step 2: Load data to PostgreSQL
    engine = get_sqlalchemy_engine()

    try:
        load_users(engine)
        load_movies(engine)
        load_genres(engine)
        load_movie_genres(engine)
        load_ratings(engine)

        logger.info("=" * 60)
        logger.info("Data loading completed successfully!")
        logger.info("=" * 60)

        # Verify counts
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    (SELECT COUNT(*) FROM users) as users_count,
                    (SELECT COUNT(*) FROM movies) as movies_count,
                    (SELECT COUNT(*) FROM genres) as genres_count,
                    (SELECT COUNT(*) FROM movie_genres) as movie_genres_count,
                    (SELECT COUNT(*) FROM ratings) as ratings_count
            """))
            row = result.fetchone()

            logger.info("Final counts:")
            logger.info(f"  Users: {row[0]:,}")
            logger.info(f"  Movies: {row[1]:,}")
            logger.info(f"  Genres: {row[2]:,}")
            logger.info(f"  Movie-Genres: {row[3]:,}")
            logger.info(f"  Ratings: {row[4]:,}")

        return True

    except Exception as e:
        logger.error(f"Error during data loading: {e}")
        raise
    finally:
        engine.dispose()


if __name__ == "__main__":
    load_all_data()

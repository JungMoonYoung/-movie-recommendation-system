"""
Main CLI for Movie Recommendation System
영화 추천 시스템 CLI
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.recommenders.popularity import get_popular_movies
from src.recommenders.genre import get_genre_based_recommendations
from src.recommenders.similarity import get_similar_movies_for_user, get_similar_movies_for_movie
from src.recommenders.ml_based import get_ml_recommendations
from src.recommenders.hybrid import get_hybrid_recommendations
from src.db_connection import get_sqlalchemy_engine
from sqlalchemy import text
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_user_exists(user_id: int) -> bool:
    """Check if user exists in database"""
    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            query = text("SELECT COUNT(*) as count FROM users WHERE user_id = :user_id")
            result = pd.read_sql(query, conn, params={'user_id': user_id})
            return result['count'].iloc[0] > 0
    except Exception as e:
        logger.error(f"Error checking user: {e}")
        return False
    finally:
        engine.dispose()


def check_movie_exists(movie_id: int) -> bool:
    """Check if movie exists in database"""
    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            query = text("SELECT COUNT(*) as count FROM movies WHERE movie_id = :movie_id")
            result = pd.read_sql(query, conn, params={'movie_id': movie_id})
            return result['count'].iloc[0] > 0
    except Exception as e:
        logger.error(f"Error checking movie: {e}")
        return False
    finally:
        engine.dispose()


def format_recommendations(df: pd.DataFrame, algo: str) -> str:
    """Format recommendations as a table"""
    if df.empty:
        return "No recommendations found."

    # Prepare output
    output = []
    output.append("\n" + "=" * 80)
    output.append(f"RECOMMENDATIONS ({algo.upper()})")
    output.append("=" * 80)

    for idx, row in df.iterrows():
        output.append(f"\n{idx + 1}. {row['title']}")
        output.append(f"   Genres: {row.get('genres', 'N/A')}")

        # Add algorithm-specific info
        if 'weighted_rating' in row:
            output.append(f"   Weighted Rating: {row['weighted_rating']:.2f}")
        if 'avg_rating' in row:
            output.append(f"   Average Rating: {row['avg_rating']:.2f} ({int(row.get('rating_count', 0))} ratings)")
        if 'combined_score' in row:
            output.append(f"   Combined Score: {row['combined_score']:.2f}")
        if 'similarity_score' in row:
            output.append(f"   Similarity Score: {row['similarity_score']:.4f}")
        if 'recommendation_score' in row:
            output.append(f"   Recommendation Score: {row['recommendation_score']:.4f}")
            if 'based_on_count' in row:
                output.append(f"   Based on {int(row['based_on_count'])} movies you liked")

    output.append("\n" + "=" * 80)
    return "\n".join(output)


def recommend_popularity(user_id: int, n: int) -> pd.DataFrame:
    """Get popularity-based recommendations"""
    logger.info(f"Generating popularity recommendations for user {user_id}...")
    start_time = time.time()

    result = get_popular_movies(n=n, min_ratings=30)

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.3f} seconds")

    return result


def recommend_genre(user_id: int, n: int) -> pd.DataFrame:
    """Get genre-based recommendations"""
    logger.info(f"Generating genre-based recommendations for user {user_id}...")
    start_time = time.time()

    result = get_genre_based_recommendations(user_id=user_id, n=n, top_genres=3, min_movie_ratings=30)

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.3f} seconds")

    return result


def recommend_similarity_user(user_id: int, n: int) -> pd.DataFrame:
    """Get item-based CF recommendations for user"""
    logger.info(f"Generating item-based CF recommendations for user {user_id}...")
    start_time = time.time()

    result = get_similar_movies_for_user(user_id=user_id, n=n, min_rating=4.0)

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.3f} seconds")

    return result


def recommend_ml(user_id: int, n: int) -> pd.DataFrame:
    """Get ML-based recommendations (SVD)"""
    logger.info(f"Generating ML-based recommendations for user {user_id}...")
    start_time = time.time()

    result = get_ml_recommendations(user_id=user_id, n=n)

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.3f} seconds")

    return result


def recommend_hybrid(user_id: int, n: int) -> pd.DataFrame:
    """Get hybrid recommendations"""
    logger.info(f"Generating hybrid recommendations for user {user_id}...")
    start_time = time.time()

    result = get_hybrid_recommendations(user_id=user_id, n=n)

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.3f} seconds")

    return result


def get_movie_title(movie_id: int) -> str:
    """Get movie title by ID"""
    engine = get_sqlalchemy_engine()
    try:
        with engine.connect() as conn:
            query = text("SELECT title FROM movies WHERE movie_id = :movie_id")
            result = pd.read_sql(query, conn, params={'movie_id': movie_id})
            if not result.empty:
                return result['title'].iloc[0]
            return f"Movie {movie_id}"
    except Exception as e:
        logger.error(f"Error fetching movie title: {e}")
        return f"Movie {movie_id}"
    finally:
        engine.dispose()


def recommend_similarity_movie(movie_id: int, n: int) -> pd.DataFrame:
    """Get similar movies for a given movie"""
    movie_title = get_movie_title(movie_id)
    logger.info(f"Finding similar movies to: {movie_title}...")
    start_time = time.time()

    result = get_similar_movies_for_movie(movie_id=movie_id, n=n)

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.3f} seconds")

    return result


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Movie Recommendation System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get popularity-based recommendations for user 10
  python main.py --user_id 10 --algo popularity --top_n 10

  # Get genre-based recommendations for user 10
  python main.py --user_id 10 --algo genre --top_n 10

  # Get item-based CF recommendations for user 10
  python main.py --user_id 10 --algo similarity --top_n 10

  # Get ML-based recommendations for user 10
  python main.py --user_id 10 --algo ml --top_n 10

  # Get hybrid recommendations for user 10
  python main.py --user_id 10 --algo hybrid --top_n 10

  # Get similar movies to movie 1
  python main.py --movie_id 1 --algo similarity --top_n 10
        """
    )

    parser.add_argument('--user_id', type=int, help='User ID for recommendations')
    parser.add_argument('--movie_id', type=int, help='Movie ID for similar movies')
    parser.add_argument('--algo', type=str, required=True,
                        choices=['popularity', 'genre', 'similarity', 'ml', 'hybrid'],
                        help='Recommendation algorithm')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of recommendations (default: 10)')

    args = parser.parse_args()

    # Validation
    if args.movie_id is None and args.user_id is None:
        parser.error("Either --user_id or --movie_id must be provided")

    if args.movie_id is not None and args.user_id is not None:
        parser.error("Cannot provide both --user_id and --movie_id")

    if args.movie_id is not None and args.algo != 'similarity':
        parser.error("--movie_id can only be used with --algo similarity")

    # Check existence
    if args.user_id is not None:
        if not check_user_exists(args.user_id):
            logger.error(f"User {args.user_id} not found in database")
            sys.exit(1)

    if args.movie_id is not None:
        if not check_movie_exists(args.movie_id):
            logger.error(f"Movie {args.movie_id} not found in database")
            sys.exit(1)

    # Generate recommendations
    try:
        if args.movie_id is not None:
            # Similar movies
            result_df = recommend_similarity_movie(args.movie_id, args.top_n)
        else:
            # User recommendations
            if args.algo == 'popularity':
                result_df = recommend_popularity(args.user_id, args.top_n)
            elif args.algo == 'genre':
                result_df = recommend_genre(args.user_id, args.top_n)
            elif args.algo == 'similarity':
                result_df = recommend_similarity_user(args.user_id, args.top_n)
            elif args.algo == 'ml':
                result_df = recommend_ml(args.user_id, args.top_n)
            elif args.algo == 'hybrid':
                result_df = recommend_hybrid(args.user_id, args.top_n)
            else:
                logger.error(f"Unknown algorithm: {args.algo}")
                sys.exit(1)

        # Display results
        print(format_recommendations(result_df, args.algo))

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

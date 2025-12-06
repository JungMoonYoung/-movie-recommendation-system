"""
ML-based Recommendation System using Matrix Factorization
ML 기반 추천 시스템 (행렬 분해 기법)

Note: This implementation uses scipy and numpy for SVD instead of scikit-surprise
      to avoid Windows installation issues.
"""
import numpy as np
import pandas as pd
import logging
import time
import pickle
from pathlib import Path
import sys
from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.db_connection import get_sqlalchemy_engine
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatrixFactorizationRecommender:
    """
    Matrix Factorization Recommender using SVD
    행렬 분해 기반 추천 시스템
    """

    def __init__(self, n_factors: int = 50, reg_lambda: float = 0.02):
        """
        Initialize recommender

        Args:
            n_factors: Number of latent factors (rank of decomposition)
            reg_lambda: Regularization parameter
        """
        self.n_factors = n_factors
        self.reg_lambda = reg_lambda
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
        self.user_id_map = {}  # user_id -> matrix index
        self.movie_id_map = {}  # movie_id -> matrix index
        self.reverse_user_map = {}  # matrix index -> user_id
        self.reverse_movie_map = {}  # matrix index -> movie_id

    def load_training_data(self) -> Tuple[csr_matrix, dict, dict]:
        """
        Load training data from database and create rating matrix
        데이터베이스에서 학습 데이터 로드 및 평점 행렬 생성

        Returns:
            Tuple[csr_matrix, dict, dict]: (rating_matrix, user_map, movie_map)
        """
        logger.info("Loading training data from database...")

        engine = get_sqlalchemy_engine()

        try:
            with engine.connect() as conn:
                query = text("""
                    SELECT user_id, movie_id, rating
                    FROM ratings_train
                    ORDER BY user_id, movie_id
                """)

                df = pd.read_sql(query, conn)

            logger.info(f"Loaded {len(df):,} ratings")

            # Create user and movie mappings
            unique_users = df['user_id'].unique()
            unique_movies = df['movie_id'].unique()

            self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
            self.movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
            self.reverse_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}
            self.reverse_movie_map = {idx: movie_id for movie_id, idx in self.movie_id_map.items()}

            # Map IDs to indices
            df['user_idx'] = df['user_id'].map(self.user_id_map)
            df['movie_idx'] = df['movie_id'].map(self.movie_id_map)

            # Create sparse rating matrix
            n_users = len(unique_users)
            n_movies = len(unique_movies)

            logger.info(f"Creating rating matrix: {n_users} users × {n_movies} movies")

            rating_matrix = csr_matrix(
                (df['rating'].values, (df['user_idx'].values, df['movie_idx'].values)),
                shape=(n_users, n_movies)
            )

            return rating_matrix, self.user_id_map, self.movie_id_map

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
        finally:
            engine.dispose()

    def train(self, rating_matrix: csr_matrix):
        """
        Train matrix factorization model using SVD
        SVD를 사용한 행렬 분해 모델 학습

        Args:
            rating_matrix: Sparse rating matrix (users × movies)
        """
        logger.info(f"Training SVD model with {self.n_factors} factors...")
        start_time = time.time()

        # Calculate global mean
        self.global_mean = rating_matrix.data.mean()
        logger.info(f"Global mean rating: {self.global_mean:.3f}")

        # Center the ratings by subtracting mean
        centered_matrix = rating_matrix.copy()
        centered_matrix.data -= self.global_mean

        # Perform SVD (Truncated SVD for sparse matrix)
        # U: user factors (n_users × n_factors)
        # s: singular values (n_factors,)
        # Vt: item factors transposed (n_factors × n_movies)
        logger.info("Performing SVD decomposition...")
        U, s, Vt = svds(centered_matrix.astype(np.float64), k=self.n_factors)

        # Convert to dense arrays and store
        self.user_factors = U
        self.item_factors = Vt.T
        self.singular_values = s

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f} seconds")

        # Log model statistics
        logger.info(f"User factors shape: {self.user_factors.shape}")
        logger.info(f"Item factors shape: {self.item_factors.shape}")
        logger.info(f"Singular values range: [{s.min():.3f}, {s.max():.3f}]")

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair
        사용자-영화 쌍의 평점 예측

        Args:
            user_id: User ID
            movie_id: Movie ID

        Returns:
            float: Predicted rating (1.0 - 5.0)
        """
        # Check if user and movie are in training data
        if user_id not in self.user_id_map or movie_id not in self.movie_id_map:
            return self.global_mean

        user_idx = self.user_id_map[user_id]
        movie_idx = self.movie_id_map[movie_id]

        # Predict: global_mean + user_factor · item_factor
        user_vec = self.user_factors[user_idx]
        item_vec = self.item_factors[movie_idx]

        # Apply singular values
        prediction = self.global_mean + np.dot(user_vec * self.singular_values, item_vec)

        # Clip to valid rating range
        prediction = np.clip(prediction, 1.0, 5.0)

        return float(prediction)

    def recommend_for_user(self, user_id: int, n: int = 10, exclude_watched: bool = True) -> pd.DataFrame:
        """
        Generate top-N recommendations for a user
        사용자를 위한 Top-N 추천 생성

        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_watched: Whether to exclude already watched movies

        Returns:
            pd.DataFrame: Recommendations with predicted ratings
        """
        if user_id not in self.user_id_map:
            logger.warning(f"User {user_id} not in training data, returning empty recommendations")
            return pd.DataFrame()

        user_idx = self.user_id_map[user_id]

        # Get user factor
        user_vec = self.user_factors[user_idx]

        # Predict ratings for all movies
        # predictions = global_mean + U[user] @ S @ V^T
        predictions = self.global_mean + np.dot(user_vec * self.singular_values, self.item_factors.T)
        predictions = np.clip(predictions, 1.0, 5.0)

        # Get watched movies if needed
        watched_movies = set()
        if exclude_watched:
            engine = get_sqlalchemy_engine()
            try:
                with engine.connect() as conn:
                    query = text("SELECT movie_id FROM ratings_train WHERE user_id = :user_id")
                    result = pd.read_sql(query, conn, params={'user_id': user_id})
                    watched_movies = set(result['movie_id'].tolist())
            except Exception as e:
                logger.error(f"Error fetching watched movies: {e}")
            finally:
                engine.dispose()

        # Create movie-prediction pairs
        movie_predictions = []
        for movie_idx, pred_rating in enumerate(predictions):
            movie_id = self.reverse_movie_map[movie_idx]

            # Skip watched movies
            if exclude_watched and movie_id in watched_movies:
                continue

            movie_predictions.append((movie_id, pred_rating))

        # Sort by predicted rating (descending)
        movie_predictions.sort(key=lambda x: x[1], reverse=True)

        # Get top N
        top_n = movie_predictions[:n]

        # Fetch movie details
        if len(top_n) == 0:
            return pd.DataFrame()

        movie_ids = [int(m[0]) for m in top_n]  # Convert numpy.int64 to Python int
        pred_ratings = {int(m[0]): m[1] for m in top_n}

        engine = get_sqlalchemy_engine()
        try:
            with engine.connect() as conn:
                query = text("""
                    SELECT
                        m.movie_id,
                        m.title,
                        STRING_AGG(DISTINCT g.genre_name, '|' ORDER BY g.genre_name) as genres
                    FROM movies m
                    LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                    LEFT JOIN genres g ON mg.genre_id = g.genre_id
                    WHERE m.movie_id = ANY(:movie_ids)
                    GROUP BY m.movie_id, m.title
                """)

                result_df = pd.read_sql(query, conn, params={'movie_ids': movie_ids})
        except Exception as e:
            logger.error(f"Error fetching movie details: {e}")
            return pd.DataFrame()
        finally:
            engine.dispose()

        # Add predicted ratings
        result_df['predicted_rating'] = result_df['movie_id'].map(pred_ratings)

        # Sort by predicted rating
        result_df = result_df.sort_values('predicted_rating', ascending=False)

        return result_df

    def save_model(self, filepath: str):
        """Save trained model to file"""
        model_data = {
            'n_factors': self.n_factors,
            'reg_lambda': self.reg_lambda,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'singular_values': self.singular_values,
            'global_mean': self.global_mean,
            'user_id_map': self.user_id_map,
            'movie_id_map': self.movie_id_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_movie_map': self.reverse_movie_map
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.n_factors = model_data['n_factors']
        self.reg_lambda = model_data['reg_lambda']
        self.user_factors = model_data['user_factors']
        self.item_factors = model_data['item_factors']
        self.singular_values = model_data['singular_values']
        self.global_mean = model_data['global_mean']
        self.user_id_map = model_data['user_id_map']
        self.movie_id_map = model_data['movie_id_map']
        self.reverse_user_map = model_data['reverse_user_map']
        self.reverse_movie_map = model_data['reverse_movie_map']

        logger.info(f"Model loaded from {filepath}")


def train_and_save_model(n_factors: int = 50, model_path: str = 'models/svd_model.pkl'):
    """
    Train SVD model and save to file
    SVD 모델 학습 및 저장

    Args:
        n_factors: Number of latent factors
        model_path: Path to save model
    """
    logger.info("=" * 60)
    logger.info("TRAINING ML-BASED RECOMMENDER (SVD)")
    logger.info("=" * 60)

    # Initialize recommender
    recommender = MatrixFactorizationRecommender(n_factors=n_factors)

    # Load training data
    rating_matrix, user_map, movie_map = recommender.load_training_data()

    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Users: {len(user_map):,}")
    logger.info(f"  Movies: {len(movie_map):,}")
    logger.info(f"  Ratings: {rating_matrix.nnz:,}")
    logger.info(f"  Sparsity: {100 * (1 - rating_matrix.nnz / (rating_matrix.shape[0] * rating_matrix.shape[1])):.2f}%")

    # Train model
    recommender.train(rating_matrix)

    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    recommender.save_model(model_path)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)

    return recommender


def get_ml_recommendations(user_id: int, n: int = 10, model_path: str = 'models/svd_model.pkl') -> pd.DataFrame:
    """
    Get ML-based recommendations for a user
    ML 기반 추천 생성

    Args:
        user_id: User ID
        n: Number of recommendations
        model_path: Path to trained model

    Returns:
        pd.DataFrame: Recommendations
    """
    # Load model
    recommender = MatrixFactorizationRecommender()
    recommender.load_model(model_path)

    # Generate recommendations
    result = recommender.recommend_for_user(user_id, n=n)

    return result


def get_recommendations_for_evaluation(
    user_ids: List[int],
    n: int = 10,
    model_path: str = 'models/svd_model.pkl'
) -> dict:
    """
    Get recommendations for multiple users (for evaluation)
    배치 평가용 추천 생성

    Args:
        user_ids: List of user IDs
        n: Number of recommendations per user
        model_path: Path to trained model

    Returns:
        dict: {user_id: [movie_id_1, movie_id_2, ...]}
    """
    logger.info(f"Generating ML recommendations for {len(user_ids)} users...")

    # Load model once
    recommender = MatrixFactorizationRecommender()
    recommender.load_model(model_path)

    recommendations = {}
    start_time = time.time()

    for idx, user_id in enumerate(user_ids):
        try:
            result_df = recommender.recommend_for_user(user_id, n=n)
            recommendations[user_id] = result_df['movie_id'].tolist()
        except Exception as e:
            logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
            recommendations[user_id] = []

        # Progress logging
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            users_per_sec = (idx + 1) / elapsed
            remaining = len(user_ids) - (idx + 1)
            eta = remaining / users_per_sec if users_per_sec > 0 else 0

            logger.info(
                f"Progress: {idx + 1}/{len(user_ids)} users "
                f"({(idx + 1) / len(user_ids) * 100:.1f}%) | "
                f"Speed: {users_per_sec:.1f} users/sec | "
                f"ETA: {eta / 60:.1f} min"
            )

    elapsed_time = time.time() - start_time
    logger.info(f"Batch evaluation completed in {elapsed_time / 60:.1f} minutes")

    return recommendations


if __name__ == "__main__":
    # Train model
    logger.info("\n[Step 1] Training SVD model...")
    recommender = train_and_save_model(n_factors=50, model_path='models/svd_model.pkl')

    # Test recommendations
    logger.info("\n[Step 2] Testing recommendations for user 1...")
    test_recs = recommender.recommend_for_user(user_id=1, n=10)

    print("\n" + "=" * 60)
    print("SAMPLE RECOMMENDATIONS (User 1)")
    print("=" * 60)
    print(test_recs.to_string(index=False))
    print("=" * 60)

    logger.info("\n[OK] ML-based recommender ready!")

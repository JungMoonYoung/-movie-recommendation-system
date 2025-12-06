"""
Train/Test split for recommendation system evaluation
시간 기반 분리: 사용자별로 최근 20%를 test set으로 사용
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DB_CONFIG, TEST_RATIO
from src.db_connection import get_sqlalchemy_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_ratings_by_time():
    """
    시간 기반으로 ratings를 train/test로 분리

    각 사용자별로:
    - timestamp 기준 정렬
    - 최근 20%를 test set
    - 나머지 80%를 train set

    최소 평점 수: 5개 미만인 사용자는 제외 (평가 불가)
    """
    logger.info("=" * 60)
    logger.info("Starting Train/Test split...")
    logger.info("=" * 60)

    engine = get_sqlalchemy_engine()

    try:
        # Step 1: 전체 ratings 로드
        logger.info("Loading ratings from database...")
        with engine.connect() as conn:
            query = text("""
                SELECT
                    r.rating_id,
                    r.user_id,
                    r.movie_id,
                    r.rating,
                    r.timestamp
                FROM ratings r
                ORDER BY r.user_id, r.timestamp
            """)
            ratings_df = pd.read_sql(query, conn)

        logger.info(f"Loaded {len(ratings_df):,} ratings")

        # Step 2: 사용자별 평점 개수 확인
        user_counts = ratings_df.groupby('user_id').size()
        min_ratings_threshold = 5

        valid_users = user_counts[user_counts >= min_ratings_threshold].index
        logger.info(f"Users with >= {min_ratings_threshold} ratings: {len(valid_users):,}")

        # 유효한 사용자만 필터링
        ratings_df = ratings_df[ratings_df['user_id'].isin(valid_users)]
        logger.info(f"Filtered ratings: {len(ratings_df):,}")

        # Step 3: 사용자별로 train/test 분리
        train_list = []
        test_list = []

        for user_id, group in ratings_df.groupby('user_id'):
            # timestamp 기준 정렬 (이미 정렬되어 있지만 확인)
            group = group.sort_values('timestamp')

            n_ratings = len(group)
            n_test = max(1, int(n_ratings * TEST_RATIO))  # 최소 1개는 test

            # 최근 n_test개를 test로
            test_ratings = group.iloc[-n_test:]
            train_ratings = group.iloc[:-n_test]

            train_list.append(train_ratings)
            test_list.append(test_ratings)

        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)

        logger.info(f"Train set: {len(train_df):,} ratings ({len(train_df)/len(ratings_df)*100:.1f}%)")
        logger.info(f"Test set: {len(test_df):,} ratings ({len(test_df)/len(ratings_df)*100:.1f}%)")

        # Step 4: 데이터베이스에 저장
        logger.info("Saving to database...")

        with engine.connect() as conn:
            # ratings_train 테이블 초기화
            conn.execute(text("TRUNCATE TABLE ratings_train CASCADE"))
            conn.commit()
            logger.info("Truncated ratings_train")

            # ratings_test 테이블 초기화
            conn.execute(text("TRUNCATE TABLE ratings_test CASCADE"))
            conn.commit()
            logger.info("Truncated ratings_test")

            # Train 데이터 삽입
            train_df[['user_id', 'movie_id', 'rating', 'timestamp']].to_sql(
                'ratings_train',
                conn,
                if_exists='append',
                index=False,
                chunksize=10000
            )
            conn.commit()
            logger.info(f"Inserted {len(train_df):,} train ratings")

            # Test 데이터 삽입
            test_df[['user_id', 'movie_id', 'rating', 'timestamp']].to_sql(
                'ratings_test',
                conn,
                if_exists='append',
                index=False,
                chunksize=10000
            )
            conn.commit()
            logger.info(f"Inserted {len(test_df):,} test ratings")

        # Step 5: 검증
        logger.info("\nVerifying split...")
        with engine.connect() as conn:
            query = text("""
                SELECT
                    'Train' as dataset,
                    COUNT(*) as count,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT movie_id) as unique_movies,
                    AVG(rating) as avg_rating
                FROM ratings_train
                UNION ALL
                SELECT
                    'Test',
                    COUNT(*),
                    COUNT(DISTINCT user_id),
                    COUNT(DISTINCT movie_id),
                    AVG(rating)
                FROM ratings_test
            """)
            stats_df = pd.read_sql(query, conn)

        logger.info("\n=== Split Statistics ===")
        logger.info(stats_df.to_string(index=False))

        logger.info("\n" + "=" * 60)
        logger.info("Train/Test split completed successfully!")
        logger.info("=" * 60)

        return {
            'train_count': len(train_df),
            'test_count': len(test_df),
            'total_users': len(valid_users),
            'train_ratio': len(train_df) / len(ratings_df),
            'test_ratio': len(test_df) / len(ratings_df)
        }

    except Exception as e:
        logger.error(f"Error during train/test split: {e}")
        raise
    finally:
        engine.dispose()


if __name__ == "__main__":
    result = split_ratings_by_time()
    print("\nSplit completed:")
    print(f"  Train: {result['train_count']:,} ratings ({result['train_ratio']*100:.1f}%)")
    print(f"  Test: {result['test_count']:,} ratings ({result['test_ratio']*100:.1f}%)")
    print(f"  Users: {result['total_users']:,}")

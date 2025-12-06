"""
Download MovieLens 1M dataset
"""
import urllib.request
import zipfile
import logging
from pathlib import Path

from config import MOVIELENS_URL, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_movielens_data():
    """
    Download and extract MovieLens 1M dataset
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = RAW_DATA_DIR / "ml-1m.zip"
    extract_path = RAW_DATA_DIR

    # Download if not already downloaded
    if not zip_path.exists():
        logger.info(f"Downloading MovieLens 1M dataset from {MOVIELENS_URL}")
        try:
            # Set timeout to 60 seconds
            req = urllib.request.Request(MOVIELENS_URL)
            with urllib.request.urlopen(req, timeout=60) as response:
                with open(zip_path, 'wb') as out_file:
                    out_file.write(response.read())
            logger.info(f"Downloaded to {zip_path}")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    else:
        logger.info(f"Dataset already exists at {zip_path}")

    # Extract zip file
    required_files = ['users.dat', 'movies.dat', 'ratings.dat']
    all_files_exist = all((RAW_DATA_DIR / 'ml-1m' / f).exists() for f in required_files)

    if not all_files_exist:
        logger.info("Extracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            logger.info(f"Extracted to {extract_path}")
        except Exception as e:
            logger.error(f"Failed to extract dataset: {e}")
            raise
    else:
        logger.info("Dataset already extracted")

    # Verify files
    ml_1m_dir = RAW_DATA_DIR / 'ml-1m'
    for file_name in required_files:
        file_path = ml_1m_dir / file_name
        if file_path.exists():
            logger.info(f"✓ {file_name} found")
        else:
            logger.error(f"✗ {file_name} not found")
            raise FileNotFoundError(f"{file_name} not found in {ml_1m_dir}")

    logger.info("MovieLens 1M dataset is ready!")
    return ml_1m_dir


if __name__ == "__main__":
    download_movielens_data()

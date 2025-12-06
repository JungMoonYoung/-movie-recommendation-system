"""
Utility functions
"""
import time
import logging
from functools import wraps
from config import TARGET_QUERY_TIME, MAX_QUERY_TIME

logger = logging.getLogger(__name__)


def time_logger(func):
    """
    Decorator to log execution time of functions

    Args:
        func: Function to decorate

    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        # Log performance
        if elapsed_time > MAX_QUERY_TIME:
            logger.warning(
                f"{func.__name__} took {elapsed_time:.2f}s "
                f"(exceeds MAX: {MAX_QUERY_TIME}s)"
            )
        elif elapsed_time > TARGET_QUERY_TIME:
            logger.info(
                f"{func.__name__} took {elapsed_time:.2f}s "
                f"(exceeds TARGET: {TARGET_QUERY_TIME}s)"
            )
        else:
            logger.info(f"{func.__name__} took {elapsed_time:.2f}s")

        return result

    return wrapper


def validate_rating(rating):
    """
    Validate rating value

    Args:
        rating: Rating value to validate

    Returns:
        bool: True if valid, False otherwise
    """
    from config import MIN_RATING, MAX_RATING, RATING_STEP

    if rating is None:
        return False

    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return False

    if not (MIN_RATING <= rating <= MAX_RATING):
        return False

    # Check if rating is a multiple of RATING_STEP using round
    remainder = round((rating - MIN_RATING) / RATING_STEP) * RATING_STEP
    if abs(rating - (MIN_RATING + remainder)) > 0.001:
        return False

    return True


def validate_top_n(n):
    """
    Validate Top-N parameter

    Args:
        n: Top-N value

    Returns:
        int: Valid Top-N value (clamped to valid range)
    """
    from config import DEFAULT_TOP_N, MAX_TOP_N

    if n is None:
        return DEFAULT_TOP_N

    if not isinstance(n, int) or n < 1:
        logger.warning(f"Invalid Top-N: {n}, using default: {DEFAULT_TOP_N}")
        return DEFAULT_TOP_N

    if n > MAX_TOP_N:
        logger.warning(f"Top-N {n} exceeds MAX {MAX_TOP_N}, clamping to {MAX_TOP_N}")
        return MAX_TOP_N

    return n

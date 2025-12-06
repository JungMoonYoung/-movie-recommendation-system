"""
Database connection management
"""
import psycopg2
from psycopg2 import pool
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
import logging

from config import DB_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages PostgreSQL database connections using connection pooling
    """
    _connection_pool = None
    _engine = None

    @classmethod
    def initialize_pool(cls, minconn=1, maxconn=10):
        """
        Initialize the connection pool

        Args:
            minconn: Minimum number of connections in the pool
            maxconn: Maximum number of connections in the pool
        """
        try:
            if cls._connection_pool is None:
                cls._connection_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn,
                    maxconn,
                    host=DB_CONFIG['host'],
                    port=DB_CONFIG['port'],
                    database=DB_CONFIG['database'],
                    user=DB_CONFIG['user'],
                    password=DB_CONFIG['password']
                )
                logger.info("Database connection pool created successfully")
        except psycopg2.Error as e:
            logger.error(f"Error creating connection pool: {e}")
            raise

    @classmethod
    def get_connection(cls):
        """
        Get a connection from the pool

        Returns:
            psycopg2.connection: Database connection
        """
        if cls._connection_pool is None:
            cls.initialize_pool()

        try:
            connection = cls._connection_pool.getconn()
            return connection
        except psycopg2.Error as e:
            logger.error(f"Error getting connection from pool: {e}")
            raise

    @classmethod
    def return_connection(cls, connection):
        """
        Return a connection to the pool

        Args:
            connection: Database connection to return
        """
        if cls._connection_pool:
            cls._connection_pool.putconn(connection)

    @classmethod
    @contextmanager
    def get_cursor(cls):
        """
        Context manager for database cursor

        Yields:
            psycopg2.cursor: Database cursor
        """
        connection = cls.get_connection()
        cursor = connection.cursor()
        try:
            yield cursor
            connection.commit()
        except Exception as e:
            connection.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()
            cls.return_connection(connection)

    @classmethod
    def get_engine(cls):
        """
        Get SQLAlchemy engine for pandas integration

        Returns:
            sqlalchemy.engine.Engine: SQLAlchemy engine
        """
        if cls._engine is None:
            connection_string = (
                f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
                f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
            )
            cls._engine = create_engine(connection_string, poolclass=NullPool)
            logger.info("SQLAlchemy engine created successfully")

        return cls._engine

    @classmethod
    def close_all_connections(cls):
        """
        Close all connections in the pool
        """
        if cls._connection_pool:
            cls._connection_pool.closeall()
            cls._connection_pool = None
            logger.info("All database connections closed")

        if cls._engine:
            cls._engine.dispose()
            cls._engine = None
            logger.info("SQLAlchemy engine disposed")


def test_connection():
    """
    Test database connection

    Returns:
        bool: True if connection is successful
    """
    try:
        with DatabaseConnection.get_cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            logger.info(f"PostgreSQL version: {version[0]}")
            return True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


# Convenience functions
def get_db_connection():
    """Get a database connection from the pool"""
    return DatabaseConnection.get_connection()


def get_sqlalchemy_engine():
    """Get SQLAlchemy engine"""
    return DatabaseConnection.get_engine()


if __name__ == "__main__":
    # Test connection
    if test_connection():
        print("Database connection successful!")
    else:
        print("Database connection failed!")

    DatabaseConnection.close_all_connections()

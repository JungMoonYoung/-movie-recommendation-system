"""
Database setup script
데이터베이스 및 스키마 초기화
"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import DB_CONFIG, PROJECT_ROOT


def create_database():
    """
    데이터베이스 및 사용자 생성
    """
    # PostgreSQL superuser로 연결 (기본 postgres 데이터베이스)
    try:
        postgres_password = input("Enter PostgreSQL superuser (postgres) password: ")
        conn_string = f"host={DB_CONFIG['host']} port={DB_CONFIG['port']} dbname=postgres user=postgres password={postgres_password}"
        conn = psycopg2.connect(conn_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # 1. 사용자 생성
        cursor.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT FROM pg_user WHERE usename = '{DB_CONFIG['user']}') THEN
                    CREATE USER {DB_CONFIG['user']} WITH PASSWORD '{DB_CONFIG['password']}';
                    RAISE NOTICE 'User {DB_CONFIG['user']} created';
                ELSE
                    RAISE NOTICE 'User {DB_CONFIG['user']} already exists';
                END IF;
            END
            $$;
        """)
        logger.info(f"User {DB_CONFIG['user']} ready")

        # 2. 데이터베이스 존재 확인
        cursor.execute(f"""
            SELECT 1 FROM pg_database WHERE datname = '{DB_CONFIG['database']}'
        """)
        exists = cursor.fetchone()

        if not exists:
            # 데이터베이스 생성
            cursor.execute(f"""
                CREATE DATABASE {DB_CONFIG['database']}
                    WITH
                    ENCODING = 'UTF8'
                    TEMPLATE = template0
            """)
            logger.info(f"Database {DB_CONFIG['database']} created")
        else:
            logger.info(f"Database {DB_CONFIG['database']} already exists")

        # 3. 권한 부여
        cursor.execute(f"""
            GRANT ALL PRIVILEGES ON DATABASE {DB_CONFIG['database']} TO {DB_CONFIG['user']}
        """)
        logger.info(f"Granted privileges to {DB_CONFIG['user']}")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise


def create_schema():
    """
    스키마 (테이블) 생성
    """
    schema_file = PROJECT_ROOT / "sql" / "schema.sql"

    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        return

    try:
        # Use connection string to avoid encoding issues
        conn_string = f"host={DB_CONFIG['host']} port={DB_CONFIG['port']} dbname={DB_CONFIG['database']} user={DB_CONFIG['user']} password={DB_CONFIG['password']}"
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()

        # Read and execute schema.sql
        # Try different encodings
        for encoding in ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']:
            try:
                with open(schema_file, 'r', encoding=encoding) as f:
                    schema_sql = f.read()
                logger.info(f"Successfully read schema.sql with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise Exception("Could not read schema.sql with any encoding")

        cursor.execute(schema_sql)
        conn.commit()

        logger.info("Schema created successfully")

        cursor.close()
        conn.close()

    except Exception as e:
        logger.error(f"Failed to create schema: {e}")
        raise


def test_connection():
    """
    데이터베이스 연결 테스트
    """
    try:
        conn_string = f"host={DB_CONFIG['host']} port={DB_CONFIG['port']} dbname={DB_CONFIG['database']} user={DB_CONFIG['user']} password={DB_CONFIG['password']}"
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()

        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        logger.info(f"PostgreSQL version: {version[0]}")

        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        logger.info(f"Tables in database: {[t[0] for t in tables]}")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("MovieLens Database Setup")
    print("=" * 60)

    # Step 1: Create database and user
    print("\n[Step 1] Creating database and user...")
    create_database()

    # Step 2: Create schema
    print("\n[Step 2] Creating schema (tables)...")
    create_schema()

    # Step 3: Test connection
    print("\n[Step 3] Testing connection...")
    if test_connection():
        print("\n✓ Database setup completed successfully!")
    else:
        print("\n✗ Database setup failed!")

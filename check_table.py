"""Check movie_similarities table structure"""
from src.db_connection import get_sqlalchemy_engine
from sqlalchemy import text

engine = get_sqlalchemy_engine()

try:
    with engine.connect() as conn:
        # Check if table exists
        result = conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'movie_similarities'
            ORDER BY ordinal_position
        """))

        print("Current movie_similarities table structure:")
        for row in result:
            print(f"  {row[0]}: {row[1]}")

        # Drop table if needed
        print("\nDropping existing table...")
        conn.execute(text("DROP TABLE IF EXISTS movie_similarities CASCADE"))
        conn.commit()
        print("Table dropped successfully!")

except Exception as e:
    print(f"Error: {e}")
finally:
    engine.dispose()

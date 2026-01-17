#!/usr/bin/env python3
"""
Soccer Film Analysis - Database Setup Script
Initializes the PostgreSQL database with required tables
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from config import settings


def check_postgres_connection():
    """Check if we can connect to PostgreSQL"""
    import psycopg2
    
    try:
        conn = psycopg2.connect(
            host=settings.db_host,
            port=settings.db_port,
            user=settings.db_user,
            password=settings.db_password,
            database="postgres"  # Connect to default db first
        )
        conn.close()
        return True
    except psycopg2.Error as e:
        logger.error(f"PostgreSQL connection failed: {e}")
        return False


def create_database():
    """Create the database if it doesn't exist"""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    conn = psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        user=settings.db_user,
        password=settings.db_password,
        database="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (settings.db_name,)
    )
    exists = cursor.fetchone()
    
    if not exists:
        logger.info(f"Creating database: {settings.db_name}")
        cursor.execute(f'CREATE DATABASE "{settings.db_name}"')
        logger.info("Database created successfully")
    else:
        logger.info(f"Database {settings.db_name} already exists")
    
    cursor.close()
    conn.close()


def create_tables():
    """Create all database tables"""
    from src.database.models import init_database
    
    logger.info("Creating database tables...")
    init_database()
    logger.info("Tables created successfully")


def drop_tables():
    """Drop all database tables (USE WITH CAUTION)"""
    from src.database.models import drop_database
    
    logger.warning("Dropping all database tables...")
    drop_database()
    logger.warning("Tables dropped")


def main():
    parser = argparse.ArgumentParser(
        description="Setup Soccer Film Analysis database"
    )
    parser.add_argument(
        "--reset", 
        action="store_true",
        help="Drop and recreate all tables (WARNING: deletes all data)"
    )
    parser.add_argument(
        "--check",
        action="store_true", 
        help="Only check database connection"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Soccer Film Analysis - Database Setup")
    print("=" * 60)
    print(f"Host: {settings.db_host}:{settings.db_port}")
    print(f"Database: {settings.db_name}")
    print(f"User: {settings.db_user}")
    print("=" * 60)
    
    # Check connection
    print("\n[1/3] Checking PostgreSQL connection...")
    if not check_postgres_connection():
        print("\n[FAIL] Failed to connect to PostgreSQL!")
        print("\nMake sure PostgreSQL is running and the credentials in .env are correct.")
        print("Example .env configuration:")
        print("  DB_HOST=localhost")
        print("  DB_PORT=5432")
        print("  DB_NAME=soccer_analysis")
        print("  DB_USER=postgres")
        print("  DB_PASSWORD=your_password")
        sys.exit(1)
    
    print("[OK] PostgreSQL connection successful")
    
    if args.check:
        print("\nConnection check complete.")
        return
    
    # Create database
    print("\n[2/3] Creating database...")
    try:
        create_database()
        print(f"[OK] Database '{settings.db_name}' ready")
    except Exception as e:
        print(f"[FAIL] Failed to create database: {e}")
        sys.exit(1)
    
    # Create/reset tables
    if args.reset:
        confirm = input("\n[WARNING]  WARNING: This will delete all existing data. Continue? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
        
        print("\n[3/3] Dropping and recreating tables...")
        try:
            drop_tables()
        except Exception as e:
            logger.debug(f"Drop tables error (may be expected): {e}")
    else:
        print("\n[3/3] Creating tables...")
    
    try:
        create_tables()
        print("[OK] All tables created successfully")
    except Exception as e:
        print(f"[FAIL] Failed to create tables: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("[OK] Database setup complete!")
    print("=" * 60)
    print("\nYou can now run the application with:")
    print("  python -m src.gui.main_window")
    print("\nOr run analysis from command line:")
    print("  python scripts/run_analysis.py --video path/to/video.mp4")


if __name__ == "__main__":
    main()

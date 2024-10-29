# Full file path: moneyverse/database/connection.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://user:password@localhost/db_name")

# Initialize SQLAlchemy engine and session maker
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency for getting a session, yielding it for use and closing it afterward.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

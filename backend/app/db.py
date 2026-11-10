"""Database configuration and session management for Brandwise."""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Locate the SQLite database file in the project backend directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "brandwise.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create SQLite engine with single-thread check disabled for FastAPI threads
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Create sessionmaker factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative SQLAlchemy models
Base = declarative_base()


def get_db():
    """Yield a database session and close it cleanly when finished."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
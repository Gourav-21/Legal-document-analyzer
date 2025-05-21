from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from sqlalchemy.exc import OperationalError # Import OperationalError
import os

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    name = Column(String)
    phone_no = Column(String)  # Changed from phoneno
    created_at = Column(DateTime, default=datetime.utcnow)

class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    analysis_type = Column(Text)
    analysis_result = Column(Text)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="analysis_history")

User.analysis_history = relationship("AnalysisHistory", back_populates="user")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
def check_db_connection():
    """Attempts to connect to the database and returns True if successful, False otherwise."""
    try:
        # Try to establish a connection
        connection = engine.connect()
        # If successful, close the connection and return True
        connection.close()
        print("Database connection successful.")
        return True
    except OperationalError as e:
        # If connection fails, print error and return False
        print(f"Database connection failed: {e}")
        return False
    except Exception as e:
        # Catch other potential exceptions during connection attempt
        print(f"An unexpected error occurred during DB connection check: {e}")
        return False
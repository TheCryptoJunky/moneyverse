from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="trader")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    asset = Column(String(100), nullable=False)
    amount = Column(Float, nullable=False)
    trade_type = Column(String(10), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="trades")

User.trades = relationship("Trade", order_by=Trade.id, back_populates="user")

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class Greenlist(Base):
    __tablename__ = "greenlist"
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, nullable=False)
    added_by = Column(Integer, nullable=False)
    active = Column(Boolean, default=True)

class Blacklist(Base):
    __tablename__ = "blacklist"
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, nullable=False)
    added_by = Column(Integer, nullable=False)
    reason = Column(String)
    active = Column(Boolean, default=True)

class Redlist(Base):
    __tablename__ = "redlist"
    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_id = Column(String, nullable=False)
    reason = Column(String)
    target_type = Column(String, default="offensive")
    active = Column(Boolean, default=True)

class Whitelist(Base):
    __tablename__ = "whitelist"
    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_id = Column(String, nullable=False)
    added_by = Column(Integer, nullable=False)
    active = Column(Boolean, default=True)

class Pumplist(Base):
    __tablename__ = "pumplist"
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, nullable=False)
    focus_duration = Column(Integer, nullable=False)  # Duration in minutes
    added_by = Column(Integer, nullable=False)
    active = Column(Boolean, default=True)
    min_liquidity_threshold = Column(Float, default=10000.0)  # Example threshold
    boost_amount = Column(Float, default=100.0)  # Default boost amount

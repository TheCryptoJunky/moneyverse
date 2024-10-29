from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

# User Table
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="trader")  # e.g., "admin", "trader"
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    trades = relationship("Trade", back_populates="user")

# Trade Table
class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    asset = Column(String(100), nullable=False)
    amount = Column(Float, nullable=False)
    trade_type = Column(String(10), nullable=False)  # e.g., "buy", "sell"
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="trades")

# Wallet and Swarm Tables
class Wallet(Base):
    __tablename__ = "wallets"
    wallet_id = Column(String, primary_key=True)
    encrypted_seed_phrase = Column(LargeBinary, nullable=True)  # Encrypted seed phrase for recovery
    balance = Column(Float, nullable=False, default=0.0)
    status = Column(String(20), nullable=False)  # e.g., "active", "inactive"
    added_to_swarm = Column(Boolean, default=False)
    creation_date = Column(DateTime, default=datetime.datetime.utcnow)
    encrypted_details = Column(LargeBinary, nullable=True)  # Encrypted sensitive details
    tokens = relationship("Token", back_populates="wallet")

class WalletSwarm(Base):
    __tablename__ = "wallet_swarm"
    id = Column(Integer, primary_key=True, autoincrement=True)
    swarm_id = Column(String(50), unique=True, nullable=False)
    wallets = Column(JSON, nullable=True)  # JSON structure for storing wallet details
    swarm_status = Column(String(20), nullable=True)  # e.g., "active", "rebalancing"
    last_rebalanced = Column(DateTime)
    encrypted_swarm_data = Column(LargeBinary, nullable=True)  # Encrypted field for sensitive data

# Token Table
class Token(Base):
    __tablename__ = "tokens"
    token_id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_id = Column(String, ForeignKey("wallets.wallet_id"))
    token_address = Column(String, nullable=False)
    balance = Column(Float, nullable=False, default=0.0)
    wallet = relationship("Wallet", back_populates="tokens")

# Greenlist, Blacklist, Redlist, Whitelist, and Pumplist Tables
class Greenlist(Base):
    __tablename__ = "greenlist"
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, nullable=False)
    added_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    active = Column(Boolean, default=True)

class Blacklist(Base):
    __tablename__ = "blacklist"
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, nullable=False)
    added_by = Column(Integer, ForeignKey("users.id"), nullable=False)
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
    added_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    active = Column(Boolean, default=True)

class Pumplist(Base):
    __tablename__ = "pumplist"
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, nullable=False)
    focus_duration = Column(Integer, nullable=False)  # Duration in minutes
    added_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    active = Column(Boolean, default=True)
    min_liquidity_threshold = Column(Float, default=10000.0)  # Example threshold
    boost_amount = Column(Float, default=100.0)  # Default boost amount

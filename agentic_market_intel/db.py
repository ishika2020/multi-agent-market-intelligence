"""Persistence layer.

Defaults to a local SQLite file so this runs with zero setup. Point
DATABASE_URL at a Postgres instance (e.g. Supabase/Neon free tier) later —
SQLAlchemy makes that a config change, not a code change.
"""

import os
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./market_intel.db")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    company = Column(String, nullable=False, index=True)
    mode = Column(String, nullable=False)  # "live" or "mock"
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)
    duration_seconds = Column(Integer)
    critic_approved = Column(Boolean)
    revise_count = Column(Integer)
    alert = Column(Boolean)
    status = Column(String)  # "approved" | "forced_unverified" | "failed"
    error = Column(Text)

    report = relationship("Report", back_populates="run", uselist=False)
    agent_calls = relationship("AgentCall", back_populates="run")


class AgentCall(Base):
    """One row per graph node execution — the per-agent latency/token/success
    breakdown that a run-level-only `Run` row can't give you."""

    __tablename__ = "agent_calls"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    node_name = Column(String, nullable=False)  # "research" | "analysis" | "insight" | "critic" | "report"
    started_at = Column(DateTime, default=datetime.utcnow)
    latency_seconds = Column(Float)
    tokens_total = Column(Integer)
    success = Column(Boolean)
    error = Column(Text)

    run = relationship("Run", back_populates="agent_calls")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    company = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    report_text = Column(Text, nullable=False)

    run = relationship("Run", back_populates="report")


def init_db():
    Base.metadata.create_all(bind=engine)

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from ..core.settings import settings


class Base(DeclarativeBase):
    """Declarative base for ORM models."""


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    confidentiality_tag: Mapped[str] = mapped_column(String(32), default="private", server_default="private")
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    meta: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=sa_text("'{}'::jsonb"))

    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "ordinal", name="uq_chunks_document_ordinal"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_line: Mapped[Optional[int]] = mapped_column(Integer)
    end_line: Mapped[Optional[int]] = mapped_column(Integer)
    page_no: Mapped[Optional[int]] = mapped_column(Integer)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(settings.vector_dim))
    meta: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, server_default=sa_text("'{}'::jsonb"))

    document: Mapped[Document] = relationship("Document", back_populates="chunks")
    contexts: Mapped[List["QAContext"]] = relationship("QAContext", back_populates="chunk", cascade="all, delete")


class QARun(Base):
    __tablename__ = "qa_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    llm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)
    template_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")

    contexts: Mapped[List["QAContext"]] = relationship(
        "QAContext", back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    answer: Mapped[Optional["QAAnswer"]] = relationship(
        "QAAnswer", back_populates="run", uselist=False, cascade="all, delete-orphan", passive_deletes=True
    )


class QAContext(Base):
    __tablename__ = "qa_contexts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("qa_runs.id", ondelete="CASCADE"), nullable=False)
    chunk_id: Mapped[Optional[int]] = mapped_column(ForeignKey("chunks.id", ondelete="SET NULL"))
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score_bm25: Mapped[Optional[float]] = mapped_column(Float)
    score_embed: Mapped[Optional[float]] = mapped_column(Float)
    score_rerank: Mapped[Optional[float]] = mapped_column(Float)
    rationale: Mapped[Optional[str]] = mapped_column(Text)

    run: Mapped[QARun] = relationship("QARun", back_populates="contexts")
    chunk: Mapped[Optional[Chunk]] = relationship("Chunk", back_populates="contexts")


class QAAnswer(Base):
    __tablename__ = "qa_answers"

    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("qa_runs.id", ondelete="CASCADE"), primary_key=True)
    answer_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)

    run: Mapped[QARun] = relationship("QARun", back_populates="answer")


engine = create_engine(settings.database_url, echo=settings.app_env == "development", pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""

    with session_scope() as session:
        yield session


__all__ = [
    "Base",
    "Document",
    "Chunk",
    "QARun",
    "QAContext",
    "QAAnswer",
    "engine",
    "SessionLocal",
    "session_scope",
    "get_session",
]

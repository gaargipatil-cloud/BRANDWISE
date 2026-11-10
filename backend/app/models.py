"""SQLAlchemy database models for drafts, findings, actions, and audit logs."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .db import Base


class Submission(Base):
    """Represents a submitted draft piece of content."""
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    channel = Column(String(64), nullable=False, default="twitter")
    status = Column(String(64), nullable=False, default="pending_review")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    versions = relationship("Version", back_populates="submission", cascade="all, delete-orphan")


class Version(Base):
    """Represents a specific revision of a submission's text."""
    __tablename__ = "versions"

    id = Column(Integer, primary_key=True, index=True)
    submission_id = Column(Integer, ForeignKey("submissions.id"), nullable=False)
    version_number = Column(Integer, nullable=False, default=1)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    submission = relationship("Submission", back_populates="versions")
    findings = relationship("Finding", back_populates="version", cascade="all, delete-orphan")
    actions = relationship("Action", back_populates="version", cascade="all, delete-orphan")


class Finding(Base):
    """Represents an issue detected by deterministic rules or LLM claim verification."""
    __tablename__ = "findings"

    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey("versions.id"), nullable=False)
    category = Column(String(64), nullable=False)
    severity = Column(String(32), nullable=False)
    original_text = Column(Text, nullable=False)
    verdict = Column(String(32), nullable=False)
    reason = Column(Text, nullable=False)
    source_name = Column(String(255), nullable=True)
    source_quote = Column(Text, nullable=True)
    suggestion = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False, default=1.0)
    status = Column(String(32), nullable=False, default="open")
    dispute_reason = Column(Text, nullable=True)

    version = relationship("Version", back_populates="findings")


class Action(Base):
    """Represents manager or creator decisions made on a version."""
    __tablename__ = "actions"

    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(Integer, ForeignKey("versions.id"), nullable=False)
    actor_role = Column(String(64), nullable=False)
    action_type = Column(String(64), nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    version = relationship("Version", back_populates="actions")


class AuditLog(Base):
    """Persistent chronological log of all meaningful system actions."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    actor_role = Column(String(64), nullable=False)
    action = Column(String(128), nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
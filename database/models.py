"""SQLAlchemy ORM models for SkillWatch AI (Phase 12).

Covers centres, rooms, cameras, students, batches, attendance,
sanctioned inventory, events, alerts, and audit logs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from database.connection import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Centre(Base):
    """Training centre entity."""
    __tablename__ = "centres"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    district = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    rooms = relationship("Room", back_populates="centre", cascade="all, delete-orphan")


class Room(Base):
    """Classroom or workshop area within a centre."""
    __tablename__ = "rooms"

    id = Column(Integer, primary_key=True, index=True)
    centre_id = Column(Integer, ForeignKey("centres.id"), nullable=True)
    room_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    room_type = Column(String(50), default="classroom")  # classroom, workshop, lab
    capacity = Column(Integer, default=30)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    centre = relationship("Centre", back_populates="rooms")
    cameras = relationship("CameraModel", back_populates="room")
    inventory = relationship("SanctionedInventory", back_populates="room")


class CameraModel(Base):
    """Camera installed in a room."""
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), unique=True, index=True, nullable=False)
    room_id = Column(String(50), ForeignKey("rooms.room_id"), nullable=True)
    source_type = Column(String(30), default="device")  # device, rtsp, file
    source = Column(String(255), nullable=True)
    status = Column(String(30), default="disconnected")  # connected, disconnected, degraded
    health = Column(String(30), default="unknown")  # normal, stale, frozen, poor_quality
    created_at = Column(DateTime, default=utc_now, nullable=False)

    room = relationship("Room", back_populates="cameras")


class StudentModel(Base):
    """Enrolled student record."""
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    batch = Column(String(50), index=True, nullable=False)
    consent_confirmed = Column(Boolean, default=True)
    sample_count = Column(Integer, default=3)
    quality_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=utc_now, nullable=False)


class BatchModel(Base):
    """Training batch / course schedule."""
    __tablename__ = "batches"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    centre_code = Column(String(50), nullable=True)
    start_time = Column(String(20), nullable=True)
    end_time = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)


class AttendanceSessionModel(Base):
    """Active or archived attendance session."""
    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, index=True)
    classroom_id = Column(String(50), index=True, nullable=False)
    batch_id = Column(String(50), index=True, nullable=False)
    session_date = Column(String(20), nullable=False)
    start_time = Column(String(20), nullable=False)
    end_time = Column(String(20), nullable=False)
    camera_id = Column(String(50), nullable=False)
    status = Column(String(20), default="CREATED")  # CREATED, ACTIVE, ENDED
    created_at = Column(DateTime, default=utc_now, nullable=False)

    records = relationship("AttendanceRecordModel", back_populates="session", cascade="all, delete-orphan")


class AttendanceRecordModel(Base):
    """Individual student attendance mark within a session."""
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=False, index=True)
    student_id = Column(String(50), index=True, nullable=False)
    status = Column(String(20), default="PRESENT")  # PRESENT, ABSENT
    confidence = Column(Float, default=1.0)
    check_in_time = Column(String(50), nullable=True)
    camera_id = Column(String(50), nullable=True)
    source = Column(String(30), default="face_recognition")  # face_recognition, manual_correction

    session = relationship("AttendanceSessionModel", back_populates="records")


class SanctionedInventory(Base):
    """Approved equipment requirements per room."""
    __tablename__ = "sanctioned_inventory"

    id = Column(Integer, primary_key=True, index=True)
    room_id = Column(String(50), ForeignKey("rooms.room_id"), nullable=False, index=True)
    asset_class = Column(String(50), nullable=False)  # chair, table, laptop, projector, etc.
    required_count = Column(Integer, default=1)
    notes = Column(Text, nullable=True)

    room = relationship("Room", back_populates="inventory")


class DetectionEventModel(Base):
    """Captured compliance or detection event."""
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(100), unique=True, index=True, nullable=False)
    camera_id = Column(String(50), index=True, nullable=False)
    event_type = Column(String(50), index=True, nullable=False)
    payload_json = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=utc_now, nullable=False)


class ComplianceAlertModel(Base):
    """Compliance alert generated by rule engine."""
    __tablename__ = "compliance_alerts_orm"

    id = Column(Integer, primary_key=True, index=True)
    fingerprint = Column(String(128), unique=True, index=True, nullable=False)
    rule_id = Column(String(80), index=True, nullable=False)
    title = Column(String(200), nullable=False)
    severity = Column(String(20), default="medium")  # low, medium, high
    status = Column(String(20), default="ACTIVE")  # ACTIVE, RESOLVED
    scope_id = Column(String(80), default="global")
    confidence = Column(Float, default=0.0)
    first_seen_at = Column(String(50), nullable=False)
    last_seen_at = Column(String(50), nullable=False)
    resolved_at = Column(String(50), nullable=True)
    consecutive_evidence = Column(Integer, default=1)
    evidence_json = Column(Text, default="{}")
    resolution_reason = Column(Text, nullable=True)


class AuditLogModel(Base):
    """Immutable audit trail for manual corrections and officer actions."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    entity_type = Column(String(50), index=True, nullable=False)  # attendance, alert, camera
    entity_id = Column(String(100), nullable=False)
    action = Column(String(50), nullable=False)
    actor = Column(String(100), default="admin")
    details_json = Column(Text, default="{}")
    timestamp = Column(DateTime, default=utc_now, nullable=False)

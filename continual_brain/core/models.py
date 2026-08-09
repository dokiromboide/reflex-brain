"""
Core data models for Reflex Brain.
SQLAlchemy models with dataclasses for type safety.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import json
import uuid


class LessonStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


class SkillStatus(str, Enum):
    DRAFT = "draft"
    TESTED = "tested"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class MemoryType(str, Enum):
    OBSERVATION = "observation"
    DECISION = "decision"
    OUTCOME = "outcome"
    PATTERN = "pattern"


class RefinementAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DEPRECATE = "deprecate"
    SUPERSEDE = "supersede"


class RefinementStatus(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class Lesson:
    """Versioned knowledge unit with evidence and confidence."""
    id: str = field(default_factory=lambda: f"lesson_{uuid.uuid4().hex[:12]}")
    version: int = 1
    title: str = ""
    content: str = ""
    evidence: list[dict] = field(default_factory=list)  # [{"source": "...", "quote": "...", "weight": 0.9}]
    confidence: float = 0.5
    status: LessonStatus = LessonStatus.PROPOSED
    supersedes_id: Optional[str] = None
    cluster_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "title": self.title,
            "content": self.content,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "status": self.status.value,
            "supersedes_id": self.supersedes_id,
            "cluster_id": self.cluster_id,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Lesson":
        lesson = cls(
            id=data["id"],
            version=data["version"],
            title=data["title"],
            content=data["content"],
            evidence=data.get("evidence", []),
            confidence=data.get("confidence", 0.5),
            status=LessonStatus(data.get("status", "proposed")),
            supersedes_id=data.get("supersedes_id"),
            cluster_id=data.get("cluster_id"),
            tags=data.get("tags", []),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat() + "Z"),
        )
        return lesson


@dataclass
class Skill:
    """Executable capability with interface, tests, and versioning."""
    id: str = field(default_factory=lambda: f"skill_{uuid.uuid4().hex[:12]}")
    version: int = 1
    name: str = ""
    description: str = ""
    code: str = ""  # Python/JS executable code
    interface: dict = field(default_factory=dict)  # {"args": [...], "returns": {...}}
    dependencies: list[str] = field(default_factory=list)
    test_cases: list[dict] = field(default_factory=list)  # [{"input": {...}, "expected": {...}}]
    confidence: float = 0.3
    status: SkillStatus = SkillStatus.DRAFT
    lesson_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "interface": self.interface,
            "dependencies": self.dependencies,
            "test_cases": self.test_cases,
            "confidence": self.confidence,
            "status": self.status.value,
            "lesson_ids": self.lesson_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        return cls(
            id=data["id"],
            version=data["version"],
            name=data["name"],
            description=data["description"],
            code=data["code"],
            interface=data.get("interface", {}),
            dependencies=data.get("dependencies", []),
            test_cases=data.get("test_cases", []),
            confidence=data.get("confidence", 0.3),
            status=SkillStatus(data.get("status", "draft")),
            lesson_ids=data.get("lesson_ids", []),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat() + "Z"),
        )


@dataclass
class Memory:
    """Episodic memory with importance decay and temporal context."""
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    type: MemoryType = MemoryType.OBSERVATION
    content: str = ""
    context: dict = field(default_factory=dict)  # {"session": "...", "goal": "...", "tools": [...]}
    importance: float = 0.5
    decay_rate: float = 0.01  # per day
    last_accessed: Optional[str] = None
    access_count: int = 0
    cluster_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "context": self.context,
            "importance": self.importance,
            "decay_rate": self.decay_rate,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "cluster_id": self.cluster_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        return cls(
            id=data["id"],
            type=MemoryType(data.get("type", "observation")),
            content=data["content"],
            context=data.get("context", {}),
            importance=data.get("importance", 0.5),
            decay_rate=data.get("decay_rate", 0.01),
            last_accessed=data.get("last_accessed"),
            access_count=data.get("access_count", 0),
            cluster_id=data.get("cluster_id"),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
        )


@dataclass
class Refinement:
    """Audit log entry for knowledge evolution."""
    id: str = field(default_factory=lambda: f"ref_{uuid.uuid4().hex[:12]}")
    target_type: str = ""  # "lesson" | "skill" | "memory"
    target_id: str = ""
    action: RefinementAction = RefinementAction.CREATE
    proposed_by: str = "agent"  # "agent" | "user" | "auto"
    evidence: list[dict] = field(default_factory=list)
    diff: dict = field(default_factory=dict)  # {field: {old: x, new: y}}
    confidence_delta: float = 0.0
    status: RefinementStatus = RefinementStatus.PENDING
    applied_at: Optional[str] = None
    rolled_back_at: Optional[str] = None
    snapshot_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "action": self.action.value,
            "proposed_by": self.proposed_by,
            "evidence": self.evidence,
            "diff": self.diff,
            "confidence_delta": self.confidence_delta,
            "status": self.status.value,
            "applied_at": self.applied_at,
            "rolled_back_at": self.rolled_back_at,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Refinement":
        return cls(
            id=data["id"],
            target_type=data["target_type"],
            target_id=data["target_id"],
            action=RefinementAction(data["action"]),
            proposed_by=data.get("proposed_by", "agent"),
            evidence=data.get("evidence", []),
            diff=data.get("diff", {}),
            confidence_delta=data.get("confidence_delta", 0.0),
            status=RefinementStatus(data.get("status", "pending")),
            applied_at=data.get("applied_at"),
            rolled_back_at=data.get("rolled_back_at"),
            snapshot_id=data.get("snapshot_id"),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
        )


@dataclass
class Snapshot:
    """Full state checkpoint for rollback."""
    id: str = field(default_factory=lambda: f"snap_{uuid.uuid4().hex[:12]}")
    label: str = ""
    state: dict = field(default_factory=dict)  # {"lessons": [...], "skills": [...], "memories": [...]}
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    trigger: str = "manual"  # "auto" | "manual" | "pre-refine" | "scheduled"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "state": self.state,
            "created_at": self.created_at,
            "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Snapshot":
        return cls(
            id=data["id"],
            label=data["label"],
            state=data.get("state", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
            trigger=data.get("trigger", "manual"),
        )
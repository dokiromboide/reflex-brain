"""
SQLite store for Reflex Brain - async, thread-safe, WAL mode.
"""
from __future__ import annotations

import json
import os
import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

from sqlalchemy import Column, Float, Integer, String, Text, delete, select, update, text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from continual_brain.core.models import (
    Lesson,
    LessonStatus,
    Memory,
    MemoryType,
    Refinement,
    RefinementAction,
    RefinementStatus,
    Skill,
    SkillStatus,
    Snapshot,
)

Base = declarative_base()


class LessonORM(Base):
    __tablename__ = "lessons"
    id = Column(String, primary_key=True)
    version = Column(Integer, default=1)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    evidence = Column(Text, nullable=False)  # JSON
    confidence = Column(Float, default=0.5)
    status = Column(SQLEnum(LessonStatus), default=LessonStatus.PROPOSED)
    supersedes_id = Column(String, nullable=True)
    cluster_id = Column(String, nullable=True)
    tags = Column(Text, nullable=False)  # JSON
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)


class SkillORM(Base):
    __tablename__ = "skills"
    id = Column(String, primary_key=True)
    version = Column(Integer, default=1)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    code = Column(Text, nullable=False)
    interface = Column(Text, nullable=False)  # JSON
    dependencies = Column(Text, nullable=False)  # JSON
    test_cases = Column(Text, nullable=False)  # JSON
    confidence = Column(Float, default=0.3)
    status = Column(SQLEnum(SkillStatus), default=SkillStatus.DRAFT)
    lesson_ids = Column(Text, nullable=False)  # JSON
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)


class MemoryORM(Base):
    __tablename__ = "memories"
    id = Column(String, primary_key=True)
    type = Column(SQLEnum(MemoryType), default=MemoryType.OBSERVATION)
    content = Column(Text, nullable=False)
    context = Column(Text, nullable=False)  # JSON
    importance = Column(Float, default=0.5)
    decay_rate = Column(Float, default=0.01)
    last_accessed = Column(String, nullable=True)
    access_count = Column(Integer, default=0)
    cluster_id = Column(String, nullable=True)
    tags = Column(Text, nullable=False, default="[]")
    created_at = Column(String, nullable=False)


class RefinementORM(Base):
    __tablename__ = "refinements"
    id = Column(String, primary_key=True)
    target_type = Column(String, nullable=False)
    target_id = Column(String, nullable=False)
    action = Column(SQLEnum(RefinementAction), nullable=False)
    proposed_by = Column(String, default="agent")
    evidence = Column(Text, nullable=False)  # JSON
    diff = Column(Text, nullable=False)  # JSON
    confidence_delta = Column(Float, default=0.0)
    status = Column(SQLEnum(RefinementStatus), default=RefinementStatus.PENDING)
    applied_at = Column(String, nullable=True)
    rolled_back_at = Column(String, nullable=True)
    snapshot_id = Column(String, nullable=True)
    created_at = Column(String, nullable=False)


class SnapshotORM(Base):
    __tablename__ = "snapshots"
    id = Column(String, primary_key=True)
    label = Column(String, nullable=False)
    state = Column(Text, nullable=False)  # JSON
    created_at = Column(String, nullable=False)
    trigger = Column(String, default="manual")


class ProposalORM(Base):
    __tablename__ = "proposals"
    id = Column(String, primary_key=True)
    payload = Column(Text, nullable=False)
    refinement_id = Column(String, nullable=True)


class SQLiteStore:
    """Async SQLite store with WAL mode for concurrent access."""

    def __init__(self, db_path: str):
        db_path = str(Path(db_path).expanduser().resolve())
        self.db_path = db_path
        self._active_session = ContextVar(f"store_session_{id(self)}", default=None)
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

        # Async engine with WAL mode
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
            connect_args={"timeout": 30},
        )
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self):
        """Create tables and enable WAL mode."""
        async with self.engine.begin() as conn:
            # Enable WAL mode for better concurrency
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            await conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
            await conn.exec_driver_sql("PRAGMA cache_size=-32768;")  # 32MB cache
            await conn.exec_driver_sql("PRAGMA temp_store=MEMORY;")
            await conn.run_sync(Base.metadata.create_all)
            # Additive, idempotent upgrade of the 0.1 schema; preserves existing data.
            columns = await conn.exec_driver_sql("PRAGMA table_info(memories)")
            if "tags" not in {row[1] for row in columns}:
                await conn.exec_driver_sql(
                    "ALTER TABLE memories ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'"
                )
            await conn.exec_driver_sql(
                "CREATE TABLE IF NOT EXISTS content_revision "
                "(id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL)"
            )
            await conn.exec_driver_sql("INSERT OR IGNORE INTO content_revision VALUES (1, 0)")
            for table in ("lessons", "skills", "memories"):
                for operation in ("INSERT", "UPDATE", "DELETE"):
                    await conn.exec_driver_sql(
                        f"CREATE TRIGGER IF NOT EXISTS revision_{table}_{operation} "
                        f"AFTER {operation} ON {table} BEGIN "
                        "UPDATE content_revision SET revision=revision+1 WHERE id=1; END"
                    )

    async def close(self):
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self):
        """Get a database session."""
        active = self._active_session.get()
        if active:
            owner, session = active
            if owner is not asyncio.current_task():
                raise RuntimeError("A transaction cannot be shared between asyncio tasks")
            yield session
            return
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    @asynccontextmanager
    async def transaction(self):
        """Serialize writers across processes and commit all nested store operations together."""
        if self._active_session.get():
            async with self.session() as session:
                yield session
            return
        async with self.session_factory() as session:
            await session.execute(text("BEGIN IMMEDIATE"))
            token = self._active_session.set((asyncio.current_task(), session))
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise
            finally:
                self._active_session.reset(token)

    async def content_revision(self):
        async with self.session() as session:
            return int((await session.execute(text(
                "SELECT revision FROM content_revision WHERE id=1"
            ))).scalar_one())

    async def export_state(self):
        """Complete consistent entity snapshot, without the former per-type row limit."""
        async with self.transaction():
            state = {}
            for name, orm, convert in (
                ("lessons", LessonORM, self._lesson_orm_to_model),
                ("skills", SkillORM, self._skill_orm_to_model),
                ("memories", MemoryORM, self._memory_orm_to_model),
            ):
                async with self.session() as session:
                    rows = (await session.execute(select(orm).order_by(orm.id))).scalars()
                    state[name] = [convert(row).to_dict() for row in rows]
            return state

    async def save_proposal(self, proposal_id, payload, refinement_id=None):
        async with self.session() as session:
            row = await session.get(ProposalORM, proposal_id)
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if row:
                if row.payload != encoded:
                    raise ValueError("Proposal ID already exists with different content")
                if refinement_id:
                    row.refinement_id = refinement_id
            else:
                session.add(ProposalORM(id=proposal_id, payload=encoded,
                                        refinement_id=refinement_id))
            await session.flush()

    async def get_proposal(self, proposal_id):
        async with self.session() as session:
            row = await session.get(ProposalORM, proposal_id)
            return {"proposal": json.loads(row.payload), "refinement_id": row.refinement_id} if row else None

    async def list_proposals(self, limit=100):
        async with self.session() as session:
            rows = (await session.execute(select(ProposalORM).limit(limit))).scalars()
            return [{"proposal": json.loads(row.payload), "refinement_id": row.refinement_id} for row in rows]

    async def delete_skill(self, skill_id):
        async with self.session() as session:
            result = await session.execute(delete(SkillORM).where(SkillORM.id == skill_id))
            return result.rowcount > 0

    # ============ Lessons ============

    async def get_lesson(self, lesson_id: str) -> Lesson | None:
        async with self.session() as session:
            result = await session.execute(
                select(LessonORM).where(LessonORM.id == lesson_id)
            )
            orm = result.scalar_one_or_none()
            return self._lesson_orm_to_model(orm) if orm else None

    async def list_lessons(
        self,
        status: LessonStatus | None = None,
        cluster_id: str | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Lesson]:
        async with self.session() as session:
            query = select(LessonORM).order_by(LessonORM.updated_at.desc())
            if status:
                query = query.where(LessonORM.status == status)
            if cluster_id:
                query = query.where(LessonORM.cluster_id == cluster_id)
            query = query.limit(limit).offset(offset)
            result = await session.execute(query)
            return [self._lesson_orm_to_model(orm) for orm in result.scalars().all()]

    async def upsert_lesson(self, lesson: Lesson) -> Lesson:
        lesson.__post_init__()
        async with self.transaction() as session:
            # Check if exists
            existing = await session.execute(
                select(LessonORM).where(LessonORM.id == lesson.id)
            )
            orm = existing.scalar_one_or_none()

            if orm:
                # Update
                orm.version = lesson.version
                orm.title = lesson.title
                orm.content = lesson.content
                orm.evidence = json.dumps(lesson.evidence, ensure_ascii=False)
                orm.confidence = lesson.confidence
                orm.status = lesson.status
                orm.supersedes_id = lesson.supersedes_id
                orm.cluster_id = lesson.cluster_id
                orm.tags = json.dumps(lesson.tags, ensure_ascii=False)
                orm.updated_at = lesson.updated_at
            else:
                # Insert
                orm = LessonORM(
                    id=lesson.id,
                    version=lesson.version,
                    title=lesson.title,
                    content=lesson.content,
                    evidence=json.dumps(lesson.evidence, ensure_ascii=False),
                    confidence=lesson.confidence,
                    status=lesson.status,
                    supersedes_id=lesson.supersedes_id,
                    cluster_id=lesson.cluster_id,
                    tags=json.dumps(lesson.tags, ensure_ascii=False),
                    created_at=lesson.created_at,
                    updated_at=lesson.updated_at,
                )
                session.add(orm)

            await session.flush()
            return lesson

    async def delete_lesson(self, lesson_id: str) -> bool:
        async with self.session() as session:
            result = await session.execute(
                delete(LessonORM).where(LessonORM.id == lesson_id)
            )
            return result.rowcount > 0

    # ============ Skills ============

    async def get_skill(self, skill_id: str) -> Skill | None:
        async with self.session() as session:
            result = await session.execute(
                select(SkillORM).where(SkillORM.id == skill_id)
            )
            orm = result.scalar_one_or_none()
            return self._skill_orm_to_model(orm) if orm else None

    async def list_skills(
        self,
        status: SkillStatus | None = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Skill]:
        async with self.session() as session:
            query = select(SkillORM).order_by(SkillORM.updated_at.desc())
            if status:
                query = query.where(SkillORM.status == status)
            query = query.limit(limit).offset(offset)
            result = await session.execute(query)
            return [self._skill_orm_to_model(orm) for orm in result.scalars().all()]

    async def upsert_skill(self, skill: Skill) -> Skill:
        skill.__post_init__()
        async with self.transaction() as session:
            existing = await session.execute(
                select(SkillORM).where(SkillORM.id == skill.id)
            )
            orm = existing.scalar_one_or_none()

            if orm:
                orm.version = skill.version
                orm.name = skill.name
                orm.description = skill.description
                orm.code = skill.code
                orm.interface = json.dumps(skill.interface, ensure_ascii=False)
                orm.dependencies = json.dumps(skill.dependencies, ensure_ascii=False)
                orm.test_cases = json.dumps(skill.test_cases, ensure_ascii=False)
                orm.confidence = skill.confidence
                orm.status = skill.status
                orm.lesson_ids = json.dumps(skill.lesson_ids, ensure_ascii=False)
                orm.updated_at = skill.updated_at
            else:
                orm = SkillORM(
                    id=skill.id,
                    version=skill.version,
                    name=skill.name,
                    description=skill.description,
                    code=skill.code,
                    interface=json.dumps(skill.interface, ensure_ascii=False),
                    dependencies=json.dumps(skill.dependencies, ensure_ascii=False),
                    test_cases=json.dumps(skill.test_cases, ensure_ascii=False),
                    confidence=skill.confidence,
                    status=skill.status,
                    lesson_ids=json.dumps(skill.lesson_ids, ensure_ascii=False),
                    created_at=skill.created_at,
                    updated_at=skill.updated_at,
                )
                session.add(orm)

            await session.flush()
            return skill

    # ============ Memories ============

    async def get_memory(self, memory_id: str) -> Memory | None:
        async with self.session() as session:
            result = await session.execute(
                select(MemoryORM).where(MemoryORM.id == memory_id)
            )
            orm = result.scalar_one_or_none()
            return self._memory_orm_to_model(orm) if orm else None

    async def list_memories(
        self,
        type: MemoryType | None = None,
        cluster_id: str | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
        offset: int = 0
    ) -> list[Memory]:
        async with self.session() as session:
            query = select(MemoryORM).order_by(MemoryORM.importance.desc())
            if type:
                query = query.where(MemoryORM.type == type)
            if cluster_id:
                query = query.where(MemoryORM.cluster_id == cluster_id)
            if min_importance > 0:
                query = query.where(MemoryORM.importance >= min_importance)
            query = query.limit(limit).offset(offset)
            result = await session.execute(query)
            return [self._memory_orm_to_model(orm) for orm in result.scalars().all()]

    async def upsert_memory(self, memory: Memory) -> Memory:
        memory.__post_init__()
        async with self.transaction() as session:
            existing = await session.execute(
                select(MemoryORM).where(MemoryORM.id == memory.id)
            )
            orm = existing.scalar_one_or_none()

            if orm:
                orm.type = memory.type
                orm.content = memory.content
                orm.context = json.dumps(memory.context, ensure_ascii=False)
                orm.importance = memory.importance
                orm.decay_rate = memory.decay_rate
                orm.last_accessed = memory.last_accessed
                orm.access_count = memory.access_count
                orm.cluster_id = memory.cluster_id
                orm.tags = json.dumps(memory.tags, ensure_ascii=False)
            else:
                orm = MemoryORM(
                    id=memory.id,
                    type=memory.type,
                    content=memory.content,
                    context=json.dumps(memory.context, ensure_ascii=False),
                    importance=memory.importance,
                    decay_rate=memory.decay_rate,
                    last_accessed=memory.last_accessed,
                    access_count=memory.access_count,
                    cluster_id=memory.cluster_id,
                    tags=json.dumps(memory.tags, ensure_ascii=False),
                    created_at=memory.created_at,
                )
                session.add(orm)

            await session.flush()
            return memory

    async def update_memory_access(self, memory_id: str):
        """Update last_accessed and increment access_count."""
        from datetime import datetime
        async with self.session() as session:
            await session.execute(
                update(MemoryORM)
                .where(MemoryORM.id == memory_id)
                .values(
                    last_accessed=datetime.utcnow().isoformat() + "Z",
                    access_count=MemoryORM.access_count + 1
                )
            )

    # ============ Refinements ============

    async def get_refinement(self, refinement_id: str) -> Refinement | None:
        async with self.session() as session:
            result = await session.execute(
                select(RefinementORM).where(RefinementORM.id == refinement_id)
            )
            orm = result.scalar_one_or_none()
            return self._refinement_orm_to_model(orm) if orm else None

    async def list_refinements(
        self,
        target_type: str | None = None,
        target_id: str | None = None,
        status: RefinementStatus | None = None,
        limit: int = 100
    ) -> list[Refinement]:
        async with self.session() as session:
            query = select(RefinementORM).order_by(RefinementORM.created_at.desc())
            if target_type:
                query = query.where(RefinementORM.target_type == target_type)
            if target_id:
                query = query.where(RefinementORM.target_id == target_id)
            if status:
                query = query.where(RefinementORM.status == status)
            query = query.limit(limit)
            result = await session.execute(query)
            return [self._refinement_orm_to_model(orm) for orm in result.scalars().all()]

    async def upsert_refinement(self, refinement: Refinement) -> Refinement:
        async with self.session() as session:
            existing = await session.execute(
                select(RefinementORM).where(RefinementORM.id == refinement.id)
            )
            orm = existing.scalar_one_or_none()

            if orm:
                orm.target_type = refinement.target_type
                orm.target_id = refinement.target_id
                orm.action = refinement.action
                orm.proposed_by = refinement.proposed_by
                orm.evidence = json.dumps(refinement.evidence, ensure_ascii=False)
                orm.diff = json.dumps(refinement.diff, ensure_ascii=False)
                orm.confidence_delta = refinement.confidence_delta
                orm.status = refinement.status
                orm.applied_at = refinement.applied_at
                orm.rolled_back_at = refinement.rolled_back_at
                orm.snapshot_id = refinement.snapshot_id
            else:
                orm = RefinementORM(
                    id=refinement.id,
                    target_type=refinement.target_type,
                    target_id=refinement.target_id,
                    action=refinement.action,
                    proposed_by=refinement.proposed_by,
                    evidence=json.dumps(refinement.evidence, ensure_ascii=False),
                    diff=json.dumps(refinement.diff, ensure_ascii=False),
                    confidence_delta=refinement.confidence_delta,
                    status=refinement.status,
                    applied_at=refinement.applied_at,
                    rolled_back_at=refinement.rolled_back_at,
                    snapshot_id=refinement.snapshot_id,
                    created_at=refinement.created_at,
                )
                session.add(orm)

            await session.flush()
            return refinement

    # ============ Snapshots ============

    async def get_snapshot(self, snapshot_id: str) -> Snapshot | None:
        async with self.session() as session:
            result = await session.execute(
                select(SnapshotORM).where(SnapshotORM.id == snapshot_id)
            )
            orm = result.scalar_one_or_none()
            return self._snapshot_orm_to_model(orm) if orm else None

    async def list_snapshots(self, limit: int = 50) -> list[Snapshot]:
        async with self.session() as session:
            query = select(SnapshotORM).order_by(SnapshotORM.created_at.desc()).limit(limit)
            result = await session.execute(query)
            return [self._snapshot_orm_to_model(orm) for orm in result.scalars().all()]

    async def create_snapshot(self, snapshot: Snapshot) -> Snapshot:
        async with self.session() as session:
            orm = SnapshotORM(
                id=snapshot.id,
                label=snapshot.label,
                state=json.dumps(snapshot.state, ensure_ascii=False),
                created_at=snapshot.created_at,
                trigger=snapshot.trigger,
            )
            session.add(orm)
            await session.flush()
            return snapshot

    # ============ Conversion helpers ============

    def _lesson_orm_to_model(self, orm: LessonORM) -> Lesson:
        return Lesson(
            id=orm.id,
            version=orm.version,
            title=orm.title,
            content=orm.content,
            evidence=json.loads(orm.evidence) if orm.evidence else [],
            confidence=orm.confidence,
            status=orm.status,
            supersedes_id=orm.supersedes_id,
            cluster_id=orm.cluster_id,
            tags=json.loads(orm.tags) if orm.tags else [],
            created_at=orm.created_at,
            updated_at=orm.updated_at,
        )

    def _skill_orm_to_model(self, orm: SkillORM) -> Skill:
        return Skill(
            id=orm.id,
            version=orm.version,
            name=orm.name,
            description=orm.description,
            code=orm.code,
            interface=json.loads(orm.interface) if orm.interface else {},
            dependencies=json.loads(orm.dependencies) if orm.dependencies else [],
            test_cases=json.loads(orm.test_cases) if orm.test_cases else [],
            confidence=orm.confidence,
            status=orm.status,
            lesson_ids=json.loads(orm.lesson_ids) if orm.lesson_ids else [],
            created_at=orm.created_at,
            updated_at=orm.updated_at,
        )

    def _memory_orm_to_model(self, orm: MemoryORM) -> Memory:
        return Memory(
            id=orm.id,
            type=orm.type,
            content=orm.content,
            context=json.loads(orm.context) if orm.context else {},
            importance=orm.importance,
            decay_rate=orm.decay_rate,
            last_accessed=orm.last_accessed,
            access_count=orm.access_count,
            cluster_id=orm.cluster_id,
            created_at=orm.created_at,
            tags=json.loads(orm.tags),
        )

    def _refinement_orm_to_model(self, orm: RefinementORM) -> Refinement:
        return Refinement(
            id=orm.id,
            target_type=orm.target_type,
            target_id=orm.target_id,
            action=orm.action,
            proposed_by=orm.proposed_by,
            evidence=json.loads(orm.evidence) if orm.evidence else [],
            diff=json.loads(orm.diff) if orm.diff else {},
            confidence_delta=orm.confidence_delta,
            status=orm.status,
            applied_at=orm.applied_at,
            rolled_back_at=orm.rolled_back_at,
            snapshot_id=orm.snapshot_id,
            created_at=orm.created_at,
        )

    def _snapshot_orm_to_model(self, orm: SnapshotORM) -> Snapshot:
        return Snapshot(
            id=orm.id,
            label=orm.label,
            state=json.loads(orm.state) if orm.state else {},
            created_at=orm.created_at,
            trigger=orm.trigger,
        )

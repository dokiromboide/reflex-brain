"""
Unit tests for SQLiteStore.
"""
import os
import tempfile

import pytest
from sqlalchemy import text

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
from continual_brain.core.store import SQLiteStore


@pytest.fixture
async def store():
    """Create a temporary store for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    store = SQLiteStore(db_path)
    await store.initialize()

    yield store

    await store.close()
    os.unlink(db_path)


class TestSQLiteStore:
    @pytest.mark.asyncio
    async def test_lesson_crud(self, store):
        lesson = Lesson(
            id="test_lesson_1",
            title="Test Lesson",
            content="Test content for lesson",
            evidence=[{"source": "test", "quote": "test", "weight": 0.8}],
            confidence=0.8,
            status=LessonStatus.ACCEPTED,
            cluster_id="test_cluster",
            tags=["test", "lesson"],
        )

        # Create
        await store.upsert_lesson(lesson)

        # Read
        fetched = await store.get_lesson("test_lesson_1")
        assert fetched is not None
        assert fetched.title == "Test Lesson"
        assert fetched.confidence == 0.8
        assert fetched.status == LessonStatus.ACCEPTED
        assert fetched.cluster_id == "test_cluster"
        assert "test" in fetched.tags

        # Update
        lesson.confidence = 0.9
        lesson.version = 2
        await store.upsert_lesson(lesson)

        fetched = await store.get_lesson("test_lesson_1")
        assert fetched.confidence == 0.9
        assert fetched.version == 2

        # List
        lessons = await store.list_lessons(limit=10)
        assert len(lessons) == 1

        # Delete
        deleted = await store.delete_lesson("test_lesson_1")
        assert deleted

        fetched = await store.get_lesson("test_lesson_1")
        assert fetched is None

    @pytest.mark.asyncio
    async def test_skill_crud(self, store):
        skill = Skill(
            id="test_skill_1",
            name="test_skill",
            description="A test skill",
            code="def test(): return 'hello'",
            interface={"args": [{"name": "x", "type": "int"}], "returns": {"type": "str"}},
            dependencies=["requests"],
            test_cases=[{"input": {"x": 1}, "expected": "hello"}],
            confidence=0.7,
            status=SkillStatus.TESTED,
        )

        await store.upsert_skill(skill)

        fetched = await store.get_skill("test_skill_1")
        assert fetched is not None
        assert fetched.name == "test_skill"
        assert fetched.code == "def test(): return 'hello'"
        assert fetched.interface["args"][0]["name"] == "x"
        assert "requests" in fetched.dependencies
        assert len(fetched.test_cases) == 1

        skills = await store.list_skills(limit=10)
        assert len(skills) == 1

    @pytest.mark.asyncio
    async def test_memory_crud(self, store):
        memory = Memory(
            id="test_mem_1",
            content="Test memory content",
            context={"session": "sess_1", "goal": "test"},
            importance=0.8,
            type=MemoryType.DECISION,
            cluster_id="test_cluster",
        )

        await store.upsert_memory(memory)

        fetched = await store.get_memory("test_mem_1")
        assert fetched is not None
        assert fetched.content == "Test memory content"
        assert fetched.importance == 0.8
        assert fetched.type == MemoryType.DECISION
        assert fetched.context["session"] == "sess_1"

        # Update access
        await store.update_memory_access("test_mem_1")
        fetched = await store.get_memory("test_mem_1")
        assert fetched.access_count == 1
        assert fetched.last_accessed is not None

    @pytest.mark.asyncio
    async def test_refinement_crud(self, store):
        ref = Refinement(
            id="test_ref_1",
            target_type="lesson",
            target_id="lesson_123",
            action=RefinementAction.UPDATE,
            proposed_by="agent",
            evidence=[{"source": "msg_1", "quote": "test"}],
            diff={"confidence": {"old": 0.5, "new": 0.8}},
            confidence_delta=0.3,
            status=RefinementStatus.APPLIED,
        )

        await store.upsert_refinement(ref)

        fetched = await store.get_refinement("test_ref_1")
        assert fetched is not None
        assert fetched.target_type == "lesson"
        assert fetched.action == RefinementAction.UPDATE
        assert fetched.confidence_delta == 0.3

        refinements = await store.list_refinements(target_type="lesson", limit=10)
        assert len(refinements) == 1

    @pytest.mark.asyncio
    async def test_snapshot_crud(self, store):
        snap = Snapshot(
            id="test_snap_1",
            label="test snapshot",
            state={"lessons": [], "skills": []},
            trigger="manual",
        )

        await store.create_snapshot(snap)

        fetched = await store.get_snapshot("test_snap_1")
        assert fetched is not None
        assert fetched.label == "test snapshot"
        assert fetched.trigger == "manual"

        snapshots = await store.list_snapshots(limit=10)
        assert len(snapshots) == 1

    @pytest.mark.asyncio
    async def test_wal_mode(self, store):
        """Verify WAL mode is enabled."""
        async with store.session() as session:
            result = await session.execute(text("PRAGMA journal_mode;"))
            mode = result.fetchone()[0]
            assert mode == "wal"

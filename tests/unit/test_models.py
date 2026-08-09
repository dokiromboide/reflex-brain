"""
Unit tests for core models.
"""

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


class TestLesson:
    def test_lesson_creation(self):
        lesson = Lesson(
            title="Test Lesson",
            content="Test content",
            evidence=[{"source": "test", "quote": "test", "weight": 0.8}],
            confidence=0.8,
            status=LessonStatus.ACCEPTED,
        )
        assert lesson.title == "Test Lesson"
        assert lesson.confidence == 0.8
        assert lesson.status == LessonStatus.ACCEPTED
        assert lesson.version == 1

    def test_lesson_serialization(self):
        lesson = Lesson(
            title="Test",
            content="Content",
            confidence=0.7,
        )
        data = lesson.to_dict()
        assert data["title"] == "Test"
        assert data["confidence"] == 0.7

        restored = Lesson.from_dict(data)
        assert restored.title == lesson.title
        assert restored.confidence == lesson.confidence


class TestSkill:
    def test_skill_creation(self):
        skill = Skill(
            name="test_skill",
            description="A test skill",
            code="def test(): pass",
            interface={"args": [], "returns": {}},
            confidence=0.7,
        )
        assert skill.name == "test_skill"
        assert skill.status == SkillStatus.DRAFT

    def test_skill_serialization(self):
        skill = Skill(name="test", description="desc", code="code")
        data = skill.to_dict()
        restored = Skill.from_dict(data)
        assert restored.name == skill.name


class TestMemory:
    def test_memory_creation(self):
        memory = Memory(
            content="Test memory",
            importance=0.8,
            type=MemoryType.DECISION,
        )
        assert memory.content == "Test memory"
        assert memory.type == MemoryType.DECISION

    def test_memory_serialization(self):
        memory = Memory(content="Test", importance=0.5)
        data = memory.to_dict()
        restored = Memory.from_dict(data)
        assert restored.content == memory.content


class TestRefinement:
    def test_refinement_creation(self):
        ref = Refinement(
            target_type="lesson",
            target_id="lesson_123",
            action=RefinementAction.UPDATE,
            evidence=[{"source": "test", "quote": "test"}],
            diff={"content": {"old": "a", "new": "b"}},
        )
        assert ref.target_type == "lesson"
        assert ref.action == RefinementAction.UPDATE
        assert ref.status == RefinementStatus.PENDING

    def test_refinement_serialization(self):
        ref = Refinement(
            target_type="skill",
            target_id="skill_123",
            action=RefinementAction.CREATE,
        )
        data = ref.to_dict()
        restored = Refinement.from_dict(data)
        assert restored.target_type == ref.target_type


class TestSnapshot:
    def test_snapshot_creation(self):
        snap = Snapshot(
            label="test snapshot",
            state={"lessons": [], "skills": []},
        )
        assert snap.label == "test snapshot"
        assert snap.trigger == "manual"

    def test_snapshot_serialization(self):
        snap = Snapshot(label="test", state={})
        data = snap.to_dict()
        restored = Snapshot.from_dict(data)
        assert restored.label == snap.label

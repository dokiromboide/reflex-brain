"""
Unit tests for RefinementEngine.
"""
import os
import tempfile

import pytest

from continual_brain.core.models import (
    Lesson,
    LessonStatus,
    RefinementAction,
    RefinementStatus,
)
from continual_brain.core.refinement import EvidenceExtractor, RefinementEngine, RefinementProposal
from continual_brain.core.store import SQLiteStore
from continual_brain.query.continual_querier import ContinualFAISSManager, ContinualQuerier


@pytest.fixture
async def store():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    store = SQLiteStore(db_path)
    await store.initialize()
    yield store
    await store.close()
    os.unlink(db_path)


@pytest.fixture
async def querier(store):
    faiss_mgr = ContinualFAISSManager()
    querier = ContinualQuerier(store, faiss_mgr)
    yield querier


@pytest.fixture
async def engine(store, querier):
    return RefinementEngine(store, querier)


class TestEvidenceExtractor:
    @pytest.mark.asyncio
    async def test_extract_decision_pattern(self, store):
        extractor = EvidenceExtractor(store)
        messages = [
            {"id": "1", "role": "user", "content": "Decidí usar vLLM para el deployment"},
            {"id": "2", "role": "assistant", "content": "Buena elección, vLLM es rápido"},
        ]
        evidence = extractor.extract_from_session(messages)
        assert len(evidence) > 0
        assert any(e["type"] == "decision" for e in evidence)

    @pytest.mark.asyncio
    async def test_extract_error_correction(self, store):
        extractor = EvidenceExtractor(store)
        messages = [
            {"id": "1", "role": "user", "content": "Error al conectar con la base de datos"},
            {"id": "2", "role": "assistant", "content": "El fix fue cambiar el puerto"},
        ]
        evidence = extractor.extract_from_session(messages)
        assert any(e["type"] == "error_correction" for e in evidence)

    @pytest.mark.asyncio
    async def test_extract_preference(self, store):
        extractor = EvidenceExtractor(store)
        messages = [
            {"id": "1", "role": "user", "content": "Prefiero usar FastAPI sobre Flask"},
        ]
        evidence = extractor.extract_from_session(messages)
        assert any(e["type"] == "preference" for e in evidence)


class TestRefinementEngine:
    @pytest.mark.asyncio
    async def test_propose_lesson_create(self, engine):
        # Add some existing lesson to test update path
        existing = Lesson(
            id="lesson_dian_1",
            title="DIAN Facturación",
            content="DIAN requiere campos X, Y, Z",
            evidence=[{"source": "old", "quote": "old", "weight": 0.5}],
            confidence=0.6,
            status=LessonStatus.ACCEPTED,
            cluster_id="dian",
        )
        await engine.store.upsert_lesson(existing)

        # New evidence about DIAN
        evidence = [
            {"type": "learning_moment", "source": "msg_1", "quote": "Aprendí que DIAN también valida el campo Z", "weight": 0.8},
            {"type": "pattern", "source": "msg_2", "quote": "Siempre que facturo, el campo Z es obligatorio", "weight": 0.7},
        ]

        proposal = engine._propose_lesson_update(existing, evidence, 1.5, "session_123")

        assert proposal is not None
        assert proposal.action == RefinementAction.UPDATE
        assert proposal.target_id == "lesson_dian_1"
        assert proposal.target_version == 2
        assert proposal.evidence_weight == 1.5

    @pytest.mark.asyncio
    async def test_propose_lesson_create_new_topic(self, engine):
        evidence = [
            {"type": "learning_moment", "source": "msg_1", "quote": "Aprendí sobre nueva regulación DIAN 2024", "weight": 0.8},
            {"type": "pattern", "source": "msg_2", "quote": "La nueva norma exige campo W", "weight": 0.7},
        ]

        proposal = engine._propose_lesson_create("dian", evidence, 1.5, "session_123")

        assert proposal is not None
        assert proposal.action == RefinementAction.CREATE
        assert proposal.target_version == 1
        assert proposal.target_type == "lesson"

    @pytest.mark.asyncio
    async def test_apply_refinement_lesson(self, engine):
        # Create a proposal
        proposal = RefinementProposal(
            action=RefinementAction.CREATE,
            target_type="lesson",
            target_id="lesson_new_1",
            target_version=1,
            diff={
                "content": {"old": None, "new": "New lesson content"},
                "confidence": {"old": 0.0, "new": 0.6},
            },
            justification="Test proposal",
            evidence=[{"source": "test", "weight": 0.8}],
            evidence_weight=0.8,
            confidence_delta=0.6,
        )

        success, ref_id = await engine.apply_refinement(proposal, auto_apply=True)

        assert success
        assert ref_id is not None

        # Verify lesson was created
        lesson = await engine.store.get_lesson("lesson_new_1")
        assert lesson is not None
        assert lesson.content == "New lesson content"
        assert lesson.confidence == 0.6

        # Verify refinement was logged
        refinement = await engine.store.get_refinement(ref_id)
        assert refinement is not None
        assert refinement.status == RefinementStatus.APPLIED

    @pytest.mark.asyncio
    async def test_rollback(self, engine):
        # First apply a refinement
        proposal = RefinementProposal(
            action=RefinementAction.CREATE,
            target_type="lesson",
            target_id="lesson_rollback_test",
            target_version=1,
            diff={
                "content": {"old": None, "new": "Content before rollback"},
                "confidence": {"old": 0.0, "new": 0.7},
            },
            justification="Test rollback",
            evidence=[{"source": "test", "weight": 0.9}],
            evidence_weight=0.9,
            confidence_delta=0.7,
        )

        success, ref_id = await engine.apply_refinement(proposal, auto_apply=True)
        assert success

        # Verify lesson exists
        lesson = await engine.store.get_lesson("lesson_rollback_test")
        assert lesson is not None
        original_content = lesson.content

        # Now rollback
        rollback_success = await engine.rollback(ref_id)
        assert rollback_success

        # Verify lesson is restored (deleted since it was CREATE)
        lesson = await engine.store.get_lesson("lesson_rollback_test")
        assert lesson is None

        # Verify refinement status
        refinement = await engine.store.get_refinement(ref_id)
        assert refinement.status == RefinementStatus.ROLLED_BACK

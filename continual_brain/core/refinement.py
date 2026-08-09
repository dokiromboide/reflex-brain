"""
Refinement Engine - Evidence-based knowledge evolution.
Core logic for proposing, validating, and applying refinements.
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, List, Dict
from collections import defaultdict

from continual_brain.core.models import (
    Lesson, Skill, Memory, Refinement, Snapshot,
    LessonStatus, SkillStatus, RefinementAction, RefinementStatus
)
from continual_brain.core.store import SQLiteStore
from continual_brain.query.hybrid_querier import HybridQuerier


@dataclass
class RefinementProposal:
    """A proposed refinement with justification."""
    action: RefinementAction
    target_type: str  # "lesson" | "skill" | "memory"
    target_id: str
    target_version: int
    diff: Dict[str, Dict[str, Any]]  # {field: {old: x, new: y}}
    justification: str
    evidence: List[Dict[str, Any]]
    evidence_weight: float
    confidence_delta: float


class EvidenceExtractor:
    """Extracts evidence patterns from session messages."""

    # Patterns that indicate learnable content
    LEARNING_PATTERNS = {
        "decision": [
            "decidí", "elegí", "opté por", "la mejor opción", "concluí que",
            "decided", "chose", "opted for", "concluded that"
        ],
        "error_correction": [
            "error", "falló", "bug", "fix", "corregí", "solucioné",
            "error", "failed", "bug", "fix", "corrected", "solved"
        ],
        "pattern_discovery": [
            "patrón", "siempre que", "cuando", "noté que", "observé",
            "pattern", "always when", "whenever", "noticed that", "observed"
        ],
        "tool_discovery": [
            "funciona", "no funciona", "mejor forma", "truco", "shortcut",
            "works", "doesn't work", "better way", "trick", "shortcut"
        ],
        "preference": [
            "prefiero", "me gusta", "no me gusta", "odio", "amo",
            "prefer", "like", "dislike", "hate", "love"
        ],
    }

    def __init__(self, store: SQLiteStore):
        self.store = store

    def extract_from_session(self, session_messages: List[Dict]) -> List[Dict]:
        """Extract evidence patterns from a list of messages."""
        evidence = []
        
        for msg in session_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            msg_id = msg.get("id", "")
            
            if role not in ("user", "assistant") or not content:
                continue
            
            # Score content length
            length_score = min(len(content) / 1000, 1.0)
            
            # Detect patterns
            for pattern_type, keywords in self.LEARNING_PATTERNS.items():
                for kw in keywords:
                    if kw.lower() in content.lower():
                        evidence.append({
                            "type": pattern_type,
                            "source": msg_id,
                            "quote": content[:500],
                            "keyword": kw,
                            "weight": 0.7 * length_score,
                            "role": role,
                        })
                        break  # One match per pattern type per message
        
        return evidence

    def cluster_evidence(self, evidence: List[Dict]) -> Dict[str, List[Dict]]:
        """Group evidence by inferred topic/cluster."""
        clusters = defaultdict(list)
        
        for ev in evidence:
            # Simple clustering by keywords in quote
            quote = ev.get("quote", "").lower()
            
            # Topic keywords (extendable)
            topic_map = {
                "dian": ["dian", "factura", "iva", "facturación", "compliance"],
                "agentes": ["agente", "agent", "subagent", "rlm", "harness"],
                "novela": ["novela", "personaje", "capítulo", "escena", "trama"],
                "config": ["config", "configuración", "setting", "hermes", "mcp"],
                "ingresos": ["ingreso", "income", "freelance", "workana", "cliente"],
                "graphrag": ["graphrag", "brain", "faiss", "embedding", "cluster"],
                "marketing": ["marketing", "seo", "contenido", "redes", "social"],
            }
            
            matched = False
            for topic, keywords in topic_map.items():
                if any(kw in quote for kw in keywords):
                    clusters[topic].append(ev)
                    matched = True
                    break
            
            if not matched:
                clusters["general"].append(ev)
        
        return dict(clusters)


class RefinementEngine:
    """
    Evidence-based refinement engine.
    Equivalent to prime-agent's `/refine` but for continual memory.
    """
    
    def __init__(
        self,
        store: SQLiteStore,
        querier: HybridQuerier,
        min_evidence_weight: float = 0.6,
        min_confidence_for_production: float = 0.8,
        auto_apply_threshold: float = 0.85,
    ):
        self.store = store
        self.querier = querier
        self.evidence_extractor = EvidenceExtractor(store)
        self.min_evidence_weight = min_evidence_weight
        self.min_confidence_for_production = min_confidence_for_production
        self.auto_apply_threshold = auto_apply_threshold

    async def analyze_session(self, session_id: str, session_messages: List[Dict]) -> List[RefinementProposal]:
        """
        Analyze a completed session and propose refinements.
        Returns list of proposals ready for review/application.
        """
        # 1. Extract evidence from messages
        evidence = self.evidence_extractor.extract_from_session(session_messages)
        if not evidence:
            return []
        
        # 2. Cluster evidence by topic
        clustered = self.evidence_extractor.cluster_evidence(evidence)
        
        # 3. For each cluster, propose refinements
        proposals = []
        for topic, topic_evidence in clustered.items():
            topic_proposals = await self._propose_for_topic(topic, topic_evidence, session_id)
            proposals.extend(topic_proposals)
        
        return proposals

    async def _propose_for_topic(
        self,
        topic: str,
        evidence: List[Dict],
        session_id: str
    ) -> List[RefinementProposal]:
        """Generate proposals for a specific topic cluster."""
        proposals = []
        total_weight = sum(e["weight"] for e in evidence)
        
        if total_weight < self.min_evidence_weight:
            return proposals
        
        # Search existing lessons for this topic
        existing_lessons = await self.store.list_lessons(
            cluster_id=topic,
            status=LessonStatus.ACCEPTED,
            limit=5
        )
        
        # Also search skills
        existing_skills = await self.store.list_skills(
            status=SkillStatus.PRODUCTION,
            limit=5
        )
        
        # Filter skills by lesson_ids matching topic
        relevant_skills = [
            s for s in existing_skills
            if any(topic in lid for lid in s.lesson_ids)
        ]
        
        # Propose lesson updates/creates
        if existing_lessons:
            for lesson in existing_lessons:
                proposal = self._propose_lesson_update(lesson, evidence, total_weight, session_id)
                if proposal:
                    proposals.append(proposal)
        else:
            proposal = self._propose_lesson_create(topic, evidence, total_weight, session_id)
            if proposal:
                proposals.append(proposal)
        
        # Propose skill creates from strong patterns
        skill_proposals = await self._propose_skills_from_evidence(topic, evidence, session_id)
        proposals.extend(skill_proposals)
        
        return proposals

    def _propose_lesson_update(
        self,
        lesson: Lesson,
        evidence: List[Dict],
        total_weight: float,
        session_id: str
    ) -> Optional[RefinementProposal]:
        """Propose an update to an existing lesson."""
        # Synthesize new content (in real impl, use LLM)
        new_content = self._synthesize_lesson_content(lesson.content, evidence)
        new_evidence = lesson.evidence + evidence
        new_confidence = min(1.0, lesson.confidence + 0.1 * len(evidence))
        
        diff = {
            "content": {"old": lesson.content, "new": new_content},
            "confidence": {"old": lesson.confidence, "new": new_confidence},
            "evidence": {"old": lesson.evidence, "new": new_evidence},
            "updated_at": {"old": lesson.updated_at, "new": datetime.utcnow().isoformat() + "Z"},
        }
        
        return RefinementProposal(
            action=RefinementAction.UPDATE,
            target_type="lesson",
            target_id=lesson.id,
            target_version=lesson.version + 1,
            diff=diff,
            justification=f"Nuevos {len(evidence)} patrones observados en sesión {session_id} refuerzan/expanden esta lección",
            evidence=evidence,
            evidence_weight=total_weight,
            confidence_delta=new_confidence - lesson.confidence,
        )

    def _propose_lesson_create(
        self,
        topic: str,
        evidence: List[Dict],
        total_weight: float,
        session_id: str
    ) -> Optional[RefinementProposal]:
        """Propose a new lesson from evidence cluster."""
        # Generate title from topic + evidence
        title = self._generate_lesson_title(topic, evidence)
        content = self._synthesize_lesson_content("", evidence)
        
        lesson_id = f"lesson_{uuid.uuid4().hex[:12]}"
        
        diff = {
            "content": {"old": None, "new": content},
            "confidence": {"old": 0.0, "new": 0.6},
            "evidence": {"old": [], "new": evidence},
        }
        
        return RefinementProposal(
            action=RefinementAction.CREATE,
            target_type="lesson",
            target_id=lesson_id,
            target_version=1,
            diff=diff,
            justification=f"Nueva lección detectada en sesión {session_id}: {len(evidence)} patrones sobre {topic}",
            evidence=evidence,
            evidence_weight=total_weight,
            confidence_delta=0.6,
        )

    async def _propose_skills_from_evidence(
        self,
        topic: str,
        evidence: List[Dict],
        session_id: str
    ) -> List[RefinementProposal]:
        """Propose executable skills from tool usage patterns."""
        proposals = []
        
        # Look for tool usage patterns in evidence
        tool_patterns = [e for e in evidence if e.get("type") == "tool_discovery"]
        if len(tool_patterns) < 2:
            return proposals
        
        # Check if we already have a skill for this
        existing_skills = await self.store.list_skills(status=SkillStatus.PRODUCTION, limit=20)
        
        # Simple heuristic: if multiple tool discoveries, propose skill
        skill_id = f"skill_{uuid.uuid4().hex[:12]}"
        
        diff = {
            "code": {"old": None, "new": "# TODO: Generate from tool patterns"},
            "interface": {"old": None, "new": {"args": [], "returns": {}}},
        }
        
        proposals.append(RefinementProposal(
            action=RefinementAction.CREATE,
            target_type="skill",
            target_id=skill_id,
            target_version=1,
            diff=diff,
            justification=f"Patrón de uso de herramientas detectado en sesión {session_id}",
            evidence=tool_patterns,
            evidence_weight=sum(e["weight"] for e in tool_patterns),
            confidence_delta=0.3,
        ))
        
        return proposals

    def _synthesize_lesson_content(self, existing: str, evidence: List[Dict]) -> str:
        """Synthesize lesson content from existing + new evidence."""
        # In production, this would use an LLM
        # For now, append evidence summary
        if not existing:
            existing = ""
        
        summary = "\n\n---\n### Nueva evidencia (auto-generado)\n"
        for i, ev in enumerate(evidence, 1):
            summary += f"{i}. [{ev['type']}] {ev['quote'][:200]}...\n"
        
        return existing + summary

    def _generate_lesson_title(self, topic: str, evidence: List[Dict]) -> str:
        """Generate a lesson title from topic and evidence."""
        topic_titles = {
            "dian": "DIAN Facturación Electrónica",
            "agentes": "Arquitectura de Agentes RLM",
            "novela": "Técnicas de Escritura de Novela",
            "config": "Configuración de Hermes/MCP",
            "ingresos": "Estrategias de Ingreso Freelance",
            "graphrag": "Arquitectura GraphRAG Continual",
            "marketing": "Marketing para PyMES Colombianas",
        }
        base = topic_titles.get(topic, topic.title())
        return f"{base}: Lección Auto-Generada"

    # ============ Application & Rollback ============

    async def apply_refinement(
        self,
        proposal: RefinementProposal,
        auto_apply: bool = False,
        proposed_by: str = "agent"
    ) -> tuple[bool, Optional[str]]:
        """
        Apply a refinement proposal.
        Returns (success, refinement_id).
        """
        # Check threshold
        if not auto_apply and proposal.evidence_weight < self.auto_apply_threshold:
            return False, None
        
        # Create snapshot before applying
        snapshot = await self._create_pre_refinement_snapshot(proposal.target_id, proposal.target_type)
        
        # Create refinement record
        refinement = Refinement(
            id=f"ref_{uuid.uuid4().hex[:12]}",
            target_type=proposal.target_type,
            target_id=proposal.target_id,
            action=proposal.action,
            proposed_by=proposed_by,
            evidence=proposal.evidence,
            diff=proposal.diff,
            confidence_delta=proposal.confidence_delta,
            status=RefinementStatus.APPLIED,
            applied_at=datetime.utcnow().isoformat() + "Z",
            snapshot_id=snapshot.id if snapshot else None,
        )
        
        # Apply the diff based on target type
        success = await self._apply_diff(proposal)
        
        if success:
            await self.store.upsert_refinement(refinement)
            if snapshot:
                await self.store.create_snapshot(snapshot)
            return True, refinement.id
        else:
            refinement.status = RefinementStatus.REJECTED
            await self.store.upsert_refinement(refinement)
            return False, None

    async def _apply_diff(self, proposal: RefinementProposal) -> bool:
        """Apply the diff to the target entity."""
        try:
            if proposal.target_type == "lesson":
                lesson = await self.store.get_lesson(proposal.target_id)
                if not lesson and proposal.action == RefinementAction.CREATE:
                    lesson = Lesson(
                        id=proposal.target_id,
                        version=proposal.target_version,
                    )
                if lesson:
                    for field_name, change in proposal.diff.items():
                        if field_name == "content":
                            lesson.content = change["new"]
                        elif field_name == "confidence":
                            lesson.confidence = change["new"]
                        elif field_name == "evidence":
                            lesson.evidence = change["new"]
                        elif field_name == "updated_at":
                            lesson.updated_at = change["new"]
                    lesson.version = proposal.target_version
                    lesson.status = LessonStatus.ACCEPTED
                    await self.store.upsert_lesson(lesson)
                    return True
            
            elif proposal.target_type == "skill":
                skill = await self.store.get_skill(proposal.target_id)
                if not skill and proposal.action == RefinementAction.CREATE:
                    skill = Skill(
                        id=proposal.target_id,
                        version=proposal.target_version,
                    )
                if skill:
                    for field_name, change in proposal.diff.items():
                        if field_name == "code":
                            skill.code = change["new"]
                        elif field_name == "interface":
                            skill.interface = change["new"]
                    skill.version = proposal.target_version
                    skill.status = SkillStatus.TESTED
                    await self.store.upsert_skill(skill)
                    return True
            
            return False
        except Exception:
            return False

    async def _create_pre_refinement_snapshot(
        self, 
        target_id: str, 
        target_type: str
    ) -> Optional[Snapshot]:
        """Create a snapshot of relevant state before refinement."""
        # Get current state of target + related
        state = {"lessons": [], "skills": [], "memories": []}
        
        if target_type == "lesson":
            lesson = await self.store.get_lesson(target_id)
            if lesson:
                state["lessons"].append(lesson.to_dict())
        elif target_type == "skill":
            skill = await self.store.get_skill(target_id)
            if skill:
                state["skills"].append(skill.to_dict())
        
        snapshot = Snapshot(
            id=f"snap_{uuid.uuid4().hex[:12]}",
            label=f"pre-refine-{target_type}-{target_id[:8]}",
            state=state,
            trigger="pre-refine",
        )
        return snapshot

    async def rollback(self, refinement_id: str) -> bool:
        """Rollback a refinement using its snapshot."""
        refinement = await self.store.get_refinement(refinement_id)
        if not refinement or not refinement.snapshot_id:
            return False
        
        snapshot = await self.store.get_snapshot(refinement.snapshot_id)
        if not snapshot:
            return False
        
        # Restore state from snapshot
        for lesson_data in snapshot.state.get("lessons", []):
            lesson = Lesson.from_dict(lesson_data)
            await self.store.upsert_lesson(lesson)
        
        for skill_data in snapshot.state.get("skills", []):
            skill = Skill.from_dict(skill_data)
            await self.store.upsert_skill(skill)
        
        # Update refinement status
        refinement.status = RefinementStatus.ROLLED_BACK
        refinement.rolled_back_at = datetime.utcnow().isoformat() + "Z"
        await self.store.upsert_refinement(refinement)
        
        return True
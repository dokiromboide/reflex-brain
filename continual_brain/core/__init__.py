"""
Continual Brain Core - Public API
"""
from __future__ import annotations

from continual_brain.core.models import (
    Lesson, Skill, Memory, Refinement, Snapshot,
    LessonStatus, SkillStatus, MemoryType, RefinementAction, RefinementStatus,
)
from continual_brain.core.store import SQLiteStore
from continual_brain.core.refinement import RefinementEngine, RefinementProposal, EvidenceExtractor
from continual_brain.core.evidence import EntityExtractor, PatternMatcher, extract_preferences, extract_tool_usage
from continual_brain.core.migration import GraphRAGMigrator, migrate_graphrag

__all__ = [
    # Models
    "Lesson", "Skill", "Memory", "Refinement", "Snapshot",
    "LessonStatus", "SkillStatus", "MemoryType", "RefinementAction", "RefinementStatus",
    # Store
    "SQLiteStore",
    # Refinement
    "RefinementEngine", "RefinementProposal", "EvidenceExtractor",
    # Evidence
    "EntityExtractor", "PatternMatcher", "extract_preferences", "extract_tool_usage",
    # Migration
    "GraphRAGMigrator", "migrate_graphrag",
]
"""
Continual Brain Core - Public API
"""
from __future__ import annotations

from continual_brain.core.evidence import (
    EntityExtractor,
    PatternMatcher,
    extract_preferences,
    extract_tool_usage,
)
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
from continual_brain.core.refinement import EvidenceExtractor, RefinementEngine, RefinementProposal
from continual_brain.core.store import SQLiteStore
from continual_brain.core.web_researcher import (
    WebResearcher, WebSearcher, ContentExtractor, KnowledgeSynthesizer,
    WebSearcher, SearchResult, ExtractedContent, ResearchProposal,
    research_topic
)

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
    # Web Research
    "WebResearcher", "WebSearcher", "ContentExtractor", "KnowledgeSynthesizer",
    "SearchResult", "ExtractedContent", "ResearchProposal",
    "research_topic",
]

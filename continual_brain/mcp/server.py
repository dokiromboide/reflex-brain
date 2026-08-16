"""
MCP Server - Thin wrapper exposing Reflex Brain tools over MCP protocol.
Pattern 1 from hermes-mcp-extensions skill.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from continual_brain.core.models import (
    LessonStatus,
    SkillStatus,
    Snapshot,
)
from continual_brain.core.refinement import RefinementEngine
from continual_brain.core.store import SQLiteStore
from continual_brain.query.hybrid_querier import HybridQuerier
from continual_brain.core.research_scheduler import (
    ResearchScheduler,
    ResearchTask,
    ResearchTrigger,
    ScheduleFrequency,
    TriggerType,
)

# Initialize core components - LAZY
DB_PATH = os.getenv("REFLEX_DB_PATH", "continual.db")
BRAIN_NODES_DIR = os.getenv("REFLEX_BRAIN_NODES", "brain/nodes")
BRAIN_EDGES_DIR = os.getenv("REFLEX_BRAIN_EDGES", "brain/edges")
BRAIN_FAISS_INDEX = os.getenv("REFLEX_BRAIN_FAISS", "brain/brain_index.faiss")
BRAIN_FAISS_MAP = os.getenv("REFLEX_BRAIN_FAISS_MAP", "brain/brain_nodes_map.pkl")
CONTINUAL_FAISS_INDEX = os.getenv("REFLEX_CONTINUAL_FAISS", "continual_index.faiss")
CONTINUAL_FAISS_MAP = os.getenv("REFLEX_CONTINUAL_FAISS_MAP", "continual_nodes_map.pkl")

# Lazy initialization
_store: SQLiteStore | None = None
_querier: HybridQuerier | None = None
_refinement_engine: RefinementEngine | None = None
_scheduler: ResearchScheduler | None = None


def _get_store() -> SQLiteStore:
    global _store
    if _store is None:
        _store = SQLiteStore(os.getenv("REFLEX_DB_PATH", "continual.db"))
    return _store


def _get_querier() -> HybridQuerier:
    global _querier
    if _querier is None:
        from continual_brain.query.hybrid_querier import HybridQuerier
        store = _get_store()
        _querier = HybridQuerier(
            store=store,
            brain_nodes_dir=os.getenv("REFLEX_BRAIN_NODES", "brain/nodes"),
            brain_edges_dir=os.getenv("REFLEX_BRAIN_EDGES", "brain/edges"),
            brain_faiss_index=os.getenv("REFLEX_BRAIN_FAISS", "brain/brain_index.faiss"),
            brain_faiss_map=os.getenv("REFLEX_BRAIN_FAISS_MAP", "brain/brain_nodes_map.pkl"),
            continual_faiss_index=os.getenv("REFLEX_CONTINUAL_FAISS", "continual_index.faiss"),
            continual_faiss_map=os.getenv("REFLEX_CONTINUAL_FAISS_MAP", "continual_nodes_map.pkl"),
        )
    return _querier


def _get_refinement_engine() -> RefinementEngine:
    global _refinement_engine
    if _refinement_engine is None:
        from continual_brain.core.refinement import RefinementEngine
        store = _get_store()
        querier = _get_querier()
        _refinement_engine = RefinementEngine(store, querier.continual_querier)
    return _refinement_engine


def _get_scheduler() -> ResearchScheduler:
    global _scheduler
    if _scheduler is None:
        store = _get_store()
        querier = _get_querier()
        from continual_brain.core.web_researcher import WebResearcher
        web_researcher = WebResearcher(store)
        _scheduler = ResearchScheduler(
            store=store,
            hybrid_querier=_get_querier(),
            web_researcher=web_researcher,
            check_interval_seconds=300
        )
    return _scheduler


# MCP Server
server = Server("reflex-brain")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="reflex_query",
            description="Query hybrid memory (GraphRAG conversations + Continual lessons/skills/memories)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 5, "description": "Number of results"},
                    "source_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by source: conversation, knowledge, lesson, skill, memory"},
                    "expand_depth": {"type": "integer", "default": 1, "description": "Graph expansion depth"},
                    "min_confidence": {"type": "number", "default": 0.0, "description": "Minimum confidence threshold"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="reflex_propose_lesson",
            description="Propose a new lesson or update from session analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic for the lesson"},
                    "session_id": {"type": "string", "description": "Session ID to analyze"},
                    "session_messages": {"type": "array", "items": {"type": "object"}, "description": "Session messages (if not in store)"},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="reflex_apply_refinement",
            description="Apply a refinement proposal",
            inputSchema={
                "type": "object",
                "properties": {
                    "refinement_id": {"type": "string", "description": "Refinement ID to apply"},
                    "auto_apply": {"type": "boolean", "default": False, "description": "Auto-apply if evidence threshold met"},
                },
                "required": ["refinement_id"],
            },
        ),
        Tool(
            name="reflex_rollback",
            description="Rollback a refinement using its snapshot",
            inputSchema={
                "type": "object",
                "properties": {
                    "refinement_id": {"type": "string", "description": "Refinement ID to rollback"},
                },
                "required": ["refinement_id"],
            },
        ),
        Tool(
            name="reflex_snapshot",
            description="Create a manual snapshot of current state",
            inputSchema={
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Snapshot label"},
                },
                "required": ["label"],
            },
        ),
        Tool(
            name="reflex_get_lesson",
            description="Get a specific lesson by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "lesson_id": {"type": "string", "description": "Lesson ID"},
                },
                "required": ["lesson_id"],
            },
        ),
        Tool(
            name="reflex_get_skill",
            description="Get a specific skill by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "Skill ID"},
                },
                "required": ["skill_id"],
            },
        ),
        Tool(
            name="reflex_list_lessons",
            description="List lessons with optional filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Filter by status: proposed, accepted, deprecated, superseded"},
                    "cluster_id": {"type": "string", "description": "Filter by cluster"},
                    "limit": {"type": "integer", "default": 20},
                },
            },
        ),
        Tool(
            name="reflex_list_skills",
            description="List skills with optional filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "Filter by status: draft, tested, production, deprecated"},
                    "limit": {"type": "integer", "default": 20},
                },
            },
        ),
        Tool(
            name="reflex_research",
            description="Automated web research: search, extract, synthesize and store knowledge on a topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Research topic"},
                    "max_sources": {"type": "integer", "default": 10, "description": "Maximum sources to process"},
                    "create_lessons": {"type": "boolean", "default": True, "description": "Create lessons from research"},
                    "create_memories": {"type": "boolean", "default": True, "description": "Create episodic memories"},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="reflex_scheduler_add_task",
            description="Add a scheduled research task",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Task name"},
                    "topic": {"type": "string", "description": "Research topic"},
                    "frequency": {"type": "string", "enum": ["hourly", "daily", "weekly", "monthly", "custom"], "default": "daily", "description": "Schedule frequency"},
                    "cron_expression": {"type": "string", "description": "Custom cron expression (if frequency=custom)"},
                    "max_sources": {"type": "integer", "default": 10, "description": "Maximum sources per run"},
                    "create_lessons": {"type": "boolean", "default": True},
                    "create_memories": {"type": "boolean", "default": True},
                    "enabled": {"type": "boolean", "default": True},
                },
                "required": ["name", "topic"],
            },
        ),
        Tool(
            name="reflex_scheduler_remove_task",
            description="Remove a scheduled research task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to remove"},
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="reflex_scheduler_list_tasks",
            description="List scheduled research tasks",
            inputSchema={
                "type": "object",
                "properties": {
                    "enabled_only": {"type": "boolean", "default": False},
                },
            },
        ),
        Tool(
            name="reflex_scheduler_add_trigger",
            description="Add a low-coverage trigger for automatic research",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Trigger name"},
                    "topic_pattern": {"type": "string", "description": "Topic pattern (e.g., 'DIAN*', 'agente*')"},
                    "min_coverage_threshold": {"type": "number", "default": 0.3, "description": "Minimum coverage score (0-1)"},
                    "min_results_threshold": {"type": "integer", "default": 3, "description": "Minimum results threshold"},
                    "cooldown_hours": {"type": "integer", "default": 24, "description": "Cooldown between triggers"},
                    "enabled": {"type": "boolean", "default": True},
                },
                "required": ["name", "topic_pattern"],
            },
        ),
        Tool(
            name="reflex_scheduler_check_coverage",
            description="Check knowledge coverage for a topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to check"},
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="reflex_scheduler_stats",
            description="Get scheduler statistics",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == "reflex_query":
            return await _handle_query(arguments)
        if name == "reflex_research":
            return await _handle_research(arguments)
        if name == "reflex_propose_lesson":
            return await _handle_propose_lesson(arguments)
        if name == "reflex_apply_refinement":
            return await _handle_apply_refinement(arguments)
        if name == "reflex_rollback":
            return await _handle_rollback(arguments)
        if name == "reflex_snapshot":
            return await _handle_snapshot(arguments)
        if name == "reflex_get_lesson":
            return await _handle_get_lesson(arguments)
        if name == "reflex_get_skill":
            return await _handle_get_skill(arguments)
        if name == "reflex_list_lessons":
            return await _handle_list_lessons(arguments)
        if name == "reflex_list_skills":
            return await _handle_list_skills(arguments)
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_query(args: dict) -> list[TextContent]:
    querier = _get_querier()
    results = await querier.query(
        query_text=args["query"],
        top_k=args.get("top_k", 5),
        source_types=args.get("source_types"),
        expand_depth=args.get("expand_depth", 1),
        min_confidence=args.get("min_confidence", 0.0),
    )
    formatted = querier.format_results(results)
    return [TextContent(type="text", text=formatted)]


async def _handle_research(args: dict) -> list[TextContent]:
    """Handle automated web research."""
    from continual_brain.core.web_researcher import research_topic
    from continual_brain.core.store import SQLiteStore
    
    store = _get_store()
    topic = args["topic"]
    max_sources = args.get("max_sources", 10)
    create_lessons = args.get("create_lessons", True)
    create_memories = args.get("create_memories", True)
    
    # Run research
    result = await research_topic(
        store=_get_store(),
        topic=topic,
        max_sources=max_sources,
        create_lessons=create_lessons,
        create_memories=create_memories
    )
    
    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]


async def _handle_propose_lesson(args: dict) -> list[TextContent]:
    # If session_messages not provided, would need to fetch from session store
    # For now, return proposal structure
    session_messages = args.get("session_messages", [])

    if not session_messages:
        return [TextContent(type="text", text=json.dumps({
            "error": "session_messages required for proposal",
            "hint": "Provide session messages or implement session store lookup"
        }))]

    refinement_engine = _get_refinement_engine()
    proposals = await refinement_engine.analyze_session(args.get("session_id", "manual"), session_messages)

    if not proposals:
        return [TextContent(type="text", text=json.dumps({"message": "No proposals generated from session"}))]

    # Return first proposal
    proposal = proposals[0]
    return [TextContent(type="text", text=json.dumps({
        "action": proposal.action.value,
        "target_type": proposal.target_type,
        "target_id": proposal.target_id,
        "target_version": proposal.target_version,
        "diff": proposal.diff,
        "justification": proposal.justification,
        "evidence_weight": proposal.evidence_weight,
        "confidence_delta": proposal.confidence_delta,
        "evidence_count": len(proposal.evidence),
    }))]


async def _handle_apply_refinement(args: dict) -> list[TextContent]:
    # This would need the proposal object - in practice, store proposals
    return [TextContent(type="text", text=json.dumps({
        "error": "Need to store proposals first. Use reflex_propose_lesson then apply from stored proposal.",
        "hint": "Implementation: store proposals in DB with pending status"
    }))]


async def _handle_rollback(args: dict) -> list[TextContent]:
    refinement_engine = _get_refinement_engine()
    success = await refinement_engine.rollback(args["refinement_id"])
    return [TextContent(type="text", text=json.dumps({"success": success}))]


async def _handle_snapshot(args: dict) -> list[TextContent]:
    # Get current state
    store = _get_store()
    lessons = await store.list_lessons(limit=1000)
    skills = await store.list_skills(limit=1000)
    memories = await store.list_memories(limit=1000)

    import uuid

    snapshot = Snapshot(
        id=f"snap_{uuid.uuid4().hex[:12]}",
        label=args["label"],
        state={
            "lessons": [l.to_dict() for l in lessons],
            "skills": [s.to_dict() for s in skills],
            "memories": [m.to_dict() for m in memories],
        },
        trigger="manual",
    )

    await store.create_snapshot(snapshot)
    return [TextContent(type="text", text=json.dumps({"success": True, "snapshot_id": snapshot.id}))]


async def _handle_get_lesson(args: dict) -> list[TextContent]:
    store = _get_store()
    lesson = await store.get_lesson(args["lesson_id"])
    if lesson:
        return [TextContent(type="text", text=json.dumps(lesson.to_dict(), ensure_ascii=False))]
    return [TextContent(type="text", text=json.dumps({"error": "Lesson not found"}))]


async def _handle_get_skill(args: dict) -> list[TextContent]:
    store = _get_store()
    skill = await store.get_skill(args["skill_id"])
    if skill:
        return [TextContent(type="text", text=json.dumps(skill.to_dict(), ensure_ascii=False))]
    return [TextContent(type="text", text=json.dumps({"error": "Skill not found"}))]


async def _handle_list_lessons(args: dict) -> list[TextContent]:
    store = _get_store()
    status = None
    if args.get("status"):
        status = LessonStatus(args["status"])
    lessons = await store.list_lessons(status=status, cluster_id=args.get("cluster_id"), limit=args.get("limit", 20))
    return [TextContent(type="text", text=json.dumps([l.to_dict() for l in lessons], ensure_ascii=False))]


async def _handle_list_skills(args: dict) -> list[TextContent]:
    store = _get_store()
    status = None
    if args.get("status"):
        status = SkillStatus(args["status"])
    skills = await store.list_skills(status=status, limit=args.get("limit", 20))
    return [TextContent(type="text", text=json.dumps([s.to_dict() for s in skills], ensure_ascii=False))]


async def _handle_scheduler_add_task(args: dict) -> list[TextContent]:
    """Add a scheduled research task."""
    scheduler = _get_scheduler()
    task = ResearchTask(
        name=args["name"],
        topic=args["topic"],
        frequency=ScheduleFrequency(args.get("frequency", "daily")),
        cron_expression=args.get("cron_expression"),
        max_sources=args.get("max_sources", 10),
        create_lessons=args.get("create_lessons", True),
        create_memories=args.get("create_memories", True),
        enabled=args.get("enabled", True),
    )
    scheduler.add_task(task)
    return [TextContent(type="text", text=json.dumps({"success": True, "task_id": task.id, "next_run": task.next_run}, ensure_ascii=False))]


async def _handle_scheduler_remove_task(args: dict) -> list[TextContent]:
    """Remove a scheduled research task."""
    scheduler = _get_scheduler()
    success = scheduler.remove_task(args["task_id"])
    return [TextContent(type="text", text=json.dumps({"success": success}, ensure_ascii=False))]


async def _handle_scheduler_list_tasks(args: dict) -> list[TextContent]:
    """List scheduled research tasks."""
    scheduler = _get_scheduler()
    tasks = scheduler.list_tasks(enabled_only=args.get("enabled_only", False))
    return [TextContent(type="text", text=json.dumps([{
        "id": t.id,
        "name": t.name,
        "topic": t.topic,
        "frequency": t.frequency.value,
        "cron_expression": t.cron_expression,
        "max_sources": t.max_sources,
        "create_lessons": t.create_lessons,
        "create_memories": t.create_memories,
        "enabled": t.enabled,
        "last_run": t.last_run,
        "next_run": t.next_run,
        "run_count": t.run_count,
    } for t in tasks], ensure_ascii=False))]


async def _handle_scheduler_add_trigger(args: dict) -> list[TextContent]:
    """Add a low-coverage trigger."""
    scheduler = _get_scheduler()
    trigger = ResearchTrigger(
        name=args["name"],
        topic_pattern=args["topic_pattern"],
        min_coverage_threshold=args.get("min_coverage_threshold", 0.3),
        min_results_threshold=args.get("min_results_threshold", 3),
        cooldown_hours=args.get("cooldown_hours", 24),
        enabled=args.get("enabled", True),
    )
    scheduler.add_trigger(trigger)
    return [TextContent(type="text", text=json.dumps({"success": True, "trigger_id": trigger.id}, ensure_ascii=False))]


async def _handle_scheduler_check_coverage(args: dict) -> list[TextContent]:
    """Check knowledge coverage for a topic."""
    scheduler = _get_scheduler()
    coverage = await scheduler.check_coverage(args["topic"])
    return [TextContent(type="text", text=json.dumps(coverage, ensure_ascii=False, indent=2))]


async def _handle_scheduler_stats(args: dict) -> list[TextContent]:
    """Get scheduler statistics."""
    scheduler = _get_scheduler()
    stats = scheduler.get_stats()
    return [TextContent(type="text", text=json.dumps(stats, ensure_ascii=False, indent=2))]


async def main():
    """Initialize and run MCP server."""
    store = _get_store()
    await store.initialize()
    # Rebuild continual index on startup
    querier = _get_querier()
    await querier.continual_querier.rebuild_index()

    # Run server over stdio
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())

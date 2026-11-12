"""Local, persistent Reflex Brain MCP server. Stdout is reserved for JSON-RPC."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import jsonschema
from mcp.server import Server
from mcp.types import CallToolResult, TextContent, Tool

from continual_brain.core.models import LessonStatus, SkillStatus, Memory, Snapshot
from continual_brain.core.refinement import RefinementEngine, RefinementProposal
from continual_brain.core.store import SQLiteStore
from continual_brain.query.hybrid_querier import HybridQuerier
from continual_brain.core.research_scheduler import ResearchScheduler, ResearchTask, ResearchTrigger
from continual_brain.core.web_researcher import WebResearcher

logger = logging.getLogger(__name__)
_store = None
_querier = None
_refinement_engine = None
_scheduler = None
_initialize_lock = asyncio.Lock()
_initialized = False

def data_dir():
    default = Path(os.getenv("LOCALAPPDATA", str(Path.home() / ".local" / "share"))) / "ReflexBrain" / "data"
    path = Path(os.getenv("REFLEX_DATA_DIR", str(default))).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

def _get_store():
    global _store
    if _store is None:
        _store = SQLiteStore(os.getenv("REFLEX_DB_PATH", str(data_dir() / "continual.db")))
    return _store

def _get_querier():
    global _querier
    if _querier is None:
        root = Path(os.getenv("REFLEX_FAISS_PATH", str(data_dir()))).expanduser()
        _querier = HybridQuerier(
            store=_get_store(),
            brain_nodes_dir=os.getenv("REFLEX_BRAIN_NODES", str(root / "brain/nodes")),
            brain_edges_dir=os.getenv("REFLEX_BRAIN_EDGES", str(root / "brain/edges")),
            brain_faiss_index=os.getenv("REFLEX_BRAIN_FAISS", str(root / "brain/brain_index.faiss")),
            brain_faiss_map=os.getenv("REFLEX_BRAIN_FAISS_MAP", str(root / "brain/brain_nodes_map.json")),
            continual_faiss_index=os.getenv("REFLEX_CONTINUAL_FAISS", str(root / "continual_index.faiss")),
            continual_faiss_map=os.getenv("REFLEX_CONTINUAL_FAISS_MAP", str(root / "continual_nodes_map.json")),
        )
    return _querier

def _get_refinement_engine():
    global _refinement_engine
    if _refinement_engine is None:
        _refinement_engine = RefinementEngine(_get_store(), _get_querier().continual_querier)
    return _refinement_engine

def _get_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = ResearchScheduler(_get_store(), _get_querier(), WebResearcher(_get_store()),
            state_dir=os.getenv("REFLEX_SCHEDULER_STATE", str(data_dir() / "scheduler")))
    return _scheduler

async def initialize():
    global _initialized
    async with _initialize_lock:
        if not _initialized:
            await _get_store().initialize()
            _initialized = True

server = Server("reflex-brain")

@server.list_tools()
async def list_tools():
    tools = [
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
    by_name = {tool.name: tool for tool in tools}
    propose = by_name["reflex_propose_lesson"].inputSchema
    propose["properties"].update({
        "content": {"type": "string", "minLength": 1, "maxLength": 50000, "description": "Explicit lesson content, instead of session extraction"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.6},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 50},
    })
    propose["anyOf"] = [{"required": ["content"]}, {"required": ["session_messages"]}]
    propose["properties"]["session_messages"] = {
        "type": "array", "minItems": 1, "maxItems": 200,
        "items": {"type": "object", "required": ["role", "content"], "properties": {
            "id": {"type": ["string", "integer"]},
            "role": {"type": "string", "enum": ["user", "assistant"]},
            "content": {"type": "string", "maxLength": 50000}}}}
    apply = by_name["reflex_apply_refinement"]
    apply.description = "Apply a stored proposal once. Explicit approval requires approved=true; auto_apply only accepts evidence above threshold."
    apply.inputSchema["properties"]["approved"] = {"type": "boolean", "default": False,
        "description": "Explicit caller approval of the reviewed proposal"}
    apply.inputSchema["properties"]["refinement_id"]["description"] = "proposal_id returned by propose (legacy parameter name)"
    by_name["reflex_query"].inputSchema["properties"]["include_untrusted"] = {
        "type": "boolean", "default": False, "description": "Include proposed web knowledge/draft skills, explicitly labeled in metadata"}
    tools.extend([
        Tool(name="reflex_remember", description="Persist an explicitly supplied episodic memory; returns its stable ID.", inputSchema={
            "type": "object", "properties": {
                "content": {"type": "string", "minLength": 1, "maxLength": 50000},
                "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 50},
                "context": {"type": "object"}}, "required": ["content"]}),
        Tool(name="reflex_list_proposals", description="List persisted proposals and their application state.", inputSchema={
            "type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100}}}),
        Tool(name="reflex_get_refinement", description="Inspect a refinement audit record.", inputSchema={
            "type": "object", "properties": {"refinement_id": {"type": "string", "minLength": 1}}, "required": ["refinement_id"]}),
        Tool(name="reflex_status", description="Inspect version, memory counts, indexing and scheduler state without loading an embedding model.", inputSchema={
            "type": "object", "properties": {}}),
        Tool(name="reflex_scheduler_list_triggers", description="List research triggers and their IDs.", inputSchema={
            "type": "object", "properties": {}}),
        Tool(name="reflex_scheduler_remove_trigger", description="Remove a research trigger by ID.", inputSchema={
            "type": "object", "properties": {"trigger_id": {"type": "string"}}, "required": ["trigger_id"]}),
    ])
    for tool in tools:
        schema = tool.inputSchema
        schema["additionalProperties"] = False
        for key, prop in schema["properties"].items():
            if prop.get("type") == "string":
                prop.setdefault("minLength", 1)
                prop.setdefault("maxLength", 50000 if key == "query" else 1000)
            if key in ("top_k", "limit", "max_sources", "min_results_threshold"):
                prop.update(minimum=1, maximum=1000 if key == "limit" else 50)
            if key == "expand_depth":
                prop.update(minimum=0, maximum=3)
            if key in ("min_confidence", "min_coverage_threshold"):
                prop.update(minimum=0, maximum=1)
            if key == "cooldown_hours":
                prop.update(minimum=1, maximum=8760)
    return tools


def response(value, error=False):
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(value, ensure_ascii=False))],
                          isError=error)

@server.call_tool()
async def call_tool(name, arguments):
    try:
        schemas = {tool.name: tool.inputSchema for tool in await list_tools()}
        if name not in schemas:
            raise ValueError(f"Unknown tool: {name}")
        jsonschema.validate(arguments, schemas[name])
        await initialize()
        return response(await dispatch(name, arguments))
    except (ValueError, KeyError, jsonschema.ValidationError) as exc:
        return response({"error": getattr(exc, "message", str(exc)), "tool": name}, error=True)
    except Exception as exc:
        logger.exception("Tool failed: %s", name)
        return response({"error": str(exc), "tool": name}, error=True)

async def dispatch(name, args):
    store = _get_store()
    if name == "reflex_query":
        results = await _get_querier().query(args["query"], top_k=args.get("top_k", 5),
            source_types=args.get("source_types"), expand_depth=args.get("expand_depth", 1),
            min_confidence=args.get("min_confidence", 0), include_untrusted=args.get("include_untrusted", False))
        return {"results": [asdict(item) for item in results],
                "context": _get_querier().format_results(results)}
    if name == "reflex_remember":
        memory = Memory(content=args["content"], importance=args.get("importance", .5),
                        tags=args.get("tags", []), context={**args.get("context", {}), "source": "mcp_explicit"})
        await store.upsert_memory(memory)
        return {"memory_id": memory.id, "memory": memory.to_dict()}
    if name == "reflex_propose_lesson":
        engine = _get_refinement_engine()
        if "content" in args:
            evidence = [{"type": "explicit_proposal", "source": "mcp", "quote": args["content"], "weight": 0.0}]
            proposal = engine._propose_lesson_create(args["topic"], evidence, 0.0, args.get("session_id", "manual"))
            proposal.diff.update({"content": {"old": None, "new": args["content"]},
                                 "title": {"old": None, "new": args["topic"]},
                                 "confidence": {"old": 0, "new": args.get("confidence", .6)},
                                 "tags": {"old": [], "new": args.get("tags", [])}})
            proposals = [proposal]
        else:
            # Topic is a caller-specified grouping, not an ignored UI field.
            evidence = engine.evidence_extractor.extract_from_session(args["session_messages"])
            proposals = await engine._propose_for_topic(args["topic"], evidence, args.get("session_id", "manual"))
        async with store.transaction():
            for proposal in proposals:
                await store.save_proposal(proposal.id, proposal.to_dict())
        return {"proposals": [{"proposal_id": p.id, **p.to_dict()} for p in proposals],
                "proposal_id": proposals[0].id if proposals else None}
    if name == "reflex_list_proposals":
        return {"proposals": await store.list_proposals(args.get("limit", 100))}
    if name == "reflex_apply_refinement":
        saved = await store.get_proposal(args["refinement_id"])
        if not saved:
            raise ValueError("Proposal not found")
        ok, ref_id = await _get_refinement_engine().apply_refinement(
            RefinementProposal.from_dict(saved["proposal"]), auto_apply=args.get("auto_apply", False),
            approved=args.get("approved", False), proposed_by="mcp")
        if not ok:
            raise ValueError("Not approved: set approved=true after review, or meet the automatic evidence policy")
        return {"success": True, "refinement_id": ref_id}
    if name == "reflex_rollback":
        if not await _get_refinement_engine().rollback(args["refinement_id"]):
            raise ValueError("Applied refinement not found")
        return {"success": True}
    if name == "reflex_get_refinement":
        refinement = await store.get_refinement(args["refinement_id"])
        if not refinement:
            raise ValueError("Refinement not found")
        return refinement.to_dict()
    if name == "reflex_snapshot":
        async with store.transaction():
            snapshot = Snapshot(label=args["label"], state=await store.export_state())
            await store.create_snapshot(snapshot)
        return {"success": True, "snapshot_id": snapshot.id,
                "counts": {key: len(values) for key, values in snapshot.state.items()}}
    if name in ("reflex_get_lesson", "reflex_get_skill"):
        kind = name.removeprefix("reflex_get_")
        entity = await getattr(store, f"get_{kind}")(args[f"{kind}_id"])
        if not entity:
            raise ValueError(f"{kind} not found")
        return entity.to_dict()
    if name == "reflex_list_lessons":
        values = await store.list_lessons(status=LessonStatus(args["status"]) if args.get("status") else None,
            cluster_id=args.get("cluster_id"), limit=args.get("limit",20))
        return [item.to_dict() for item in values]
    if name == "reflex_list_skills":
        values = await store.list_skills(status=SkillStatus(args["status"]) if args.get("status") else None,
            limit=args.get("limit",20))
        return [item.to_dict() for item in values]
    if name == "reflex_research":
        researcher = WebResearcher(store)
        try:
            result = await researcher.research_topic(**args)
            if result.get("error"):
                raise ValueError(result["error"])
            return result
        finally:
            await researcher.close()
    if name == "reflex_status":
        from continual_brain import __version__
        state = await store.export_state()
        return {"version": __version__, "database": store.db_path,
                "counts": {key: len(values) for key, values in state.items()},
                "revision": await store.content_revision(),
                "indexing": "automatic on query; SQLite is authoritative",
                "scheduler": _scheduler.get_stats() if _scheduler else {"scheduler_running": False}}
    scheduler = _get_scheduler()
    if name == "reflex_scheduler_add_task":
        task = scheduler.add_task(ResearchTask(**args))
        return {"task_id": task.id, "next_run": task.next_run}
    if name == "reflex_scheduler_remove_task":
        return {"success": scheduler.remove_task(args["task_id"])}
    if name == "reflex_scheduler_list_tasks":
        return [asdict(task) for task in scheduler.list_tasks(args.get("enabled_only", False))]
    if name == "reflex_scheduler_add_trigger":
        trigger = scheduler.add_trigger(ResearchTrigger(**args))
        return {"trigger_id": trigger.id}
    if name == "reflex_scheduler_check_coverage":
        return await scheduler.check_coverage(args["topic"])
    if name == "reflex_scheduler_stats":
        return scheduler.get_stats()
    if name == "reflex_scheduler_list_triggers":
        return [asdict(trigger) for trigger in scheduler.list_triggers()]
    if name == "reflex_scheduler_remove_trigger":
        return {"success": scheduler.remove_trigger(args["trigger_id"])}
    raise ValueError(f"Unknown tool: {name}")

async def main():
    await initialize()
    # No model download/rebuild on startup. Empty databases answer immediately.
    if os.getenv("REFLEX_SCHEDULER_ENABLED", "0") == "1":
        await _get_scheduler().start()
    from mcp.server.stdio import stdio_server
    try:
        async with stdio_server() as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())
    finally:
        if _scheduler:
            await _scheduler.stop()
            await _scheduler.web_researcher.close()
        await _get_store().close()

def run():
    logging.basicConfig(level=logging.INFO)
    if os.name == "nt":
        # Native ML imports can block after Windows stdio reader threads start.
        # Initialize the libraries before opening that stream. Model weights and
        # the index are still loaded lazily, only for nonempty semantic queries.
        from sentence_transformers import SentenceTransformer  # noqa: F401
    asyncio.run(main())

if __name__ == "__main__":
    run()

"""
MCP Server - Thin wrapper exposing Reflex Brain tools over MCP protocol.
Pattern 1 from hermes-mcp-extensions skill.
"""
from __future__ import annotations
import os
import json
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import asdict

from mcp.server import Server
from mcp.types import Tool, TextContent

from continual_brain.core.store import SQLiteStore
from continual_brain.core.refinement import RefinementEngine, RefinementProposal
from continual_brain.query.hybrid_querier import HybridQuerier, HybridQueryResult
from continual_brain.core.models import (
    Lesson, Skill, Memory, Refinement, Snapshot,
    LessonStatus, SkillStatus, MemoryType, RefinementAction, RefinementStatus
)

# Initialize core components
DB_PATH = os.getenv("REFLEX_DB_PATH", "continual.db")
BRAIN_NODES_DIR = os.getenv("REFLEX_BRAIN_NODES", "brain/nodes")
BRAIN_EDGES_DIR = os.getenv("REFLEX_BRAIN_EDGES", "brain/edges")
BRAIN_FAISS_INDEX = os.getenv("REFLEX_BRAIN_FAISS", "brain/brain_index.faiss")
BRAIN_FAISS_MAP = os.getenv("REFLEX_BRAIN_FAISS_MAP", "brain/brain_nodes_map.pkl")
CONTINUAL_FAISS_INDEX = os.getenv("REFLEX_CONTINUAL_FAISS", "continual_index.faiss")
CONTINUAL_FAISS_MAP = os.getenv("REFLEX_CONTINUAL_FAISS_MAP", "continual_nodes_map.pkl")

store = SQLiteStore(DB_PATH)
querier = HybridQuerier(
    store=store,
    brain_nodes_dir=BRAIN_NODES_DIR,
    brain_edges_dir=BRAIN_EDGES_DIR,
    brain_faiss_index=BRAIN_FAISS_INDEX,
    brain_faiss_map=BRAIN_FAISS_MAP,
    continual_faiss_index=CONTINUAL_FAISS_INDEX,
    continual_faiss_map=CONTINUAL_FAISS_MAP,
)
refinement_engine = RefinementEngine(store, querier.continual_querier)

# MCP Server
server = Server("reflex-brain")


@server.list_tools()
async def list_tools() -> List[Tool]:
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
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    try:
        if name == "reflex_query":
            return await _handle_query(arguments)
        elif name == "reflex_propose_lesson":
            return await _handle_propose_lesson(arguments)
        elif name == "reflex_apply_refinement":
            return await _handle_apply_refinement(arguments)
        elif name == "reflex_rollback":
            return await _handle_rollback(arguments)
        elif name == "reflex_snapshot":
            return await _handle_snapshot(arguments)
        elif name == "reflex_get_lesson":
            return await _handle_get_lesson(arguments)
        elif name == "reflex_get_skill":
            return await _handle_get_skill(arguments)
        elif name == "reflex_list_lessons":
            return await _handle_list_lessons(arguments)
        elif name == "reflex_list_skills":
            return await _handle_list_skills(arguments)
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_query(args: Dict) -> List[TextContent]:
    results = await querier.query(
        query_text=args["query"],
        top_k=args.get("top_k", 5),
        source_types=args.get("source_types"),
        expand_depth=args.get("expand_depth", 1),
        min_confidence=args.get("min_confidence", 0.0),
    )
    formatted = querier.format_results(results)
    return [TextContent(type="text", text=formatted)]


async def _handle_propose_lesson(args: Dict) -> List[TextContent]:
    # If session_messages not provided, would need to fetch from session store
    # For now, return proposal structure
    session_messages = args.get("session_messages", [])
    
    if not session_messages:
        return [TextContent(type="text", text=json.dumps({
            "error": "session_messages required for proposal",
            "hint": "Provide session messages or implement session store lookup"
        }))]
    
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


async def _handle_apply_refinement(args: Dict) -> List[TextContent]:
    # This would need the proposal object - in practice, store proposals
    return [TextContent(type="text", text=json.dumps({
        "error": "Need to store proposals first. Use reflex_propose_lesson then apply from stored proposal.",
        "hint": "Implementation: store proposals in DB with pending status"
    }))]


async def _handle_rollback(args: Dict) -> List[TextContent]:
    success = await refinement_engine.rollback(args["refinement_id"])
    return [TextContent(type="text", text=json.dumps({"success": success}))]


async def _handle_snapshot(args: Dict) -> List[TextContent]:
    # Get current state
    lessons = await store.list_lessons(limit=1000)
    skills = await store.list_skills(limit=1000)
    memories = await store.list_memories(limit=1000)
    
    from continual_brain.core.models import Snapshot
    import uuid
    from datetime import datetime
    
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


async def _handle_get_lesson(args: Dict) -> List[TextContent]:
    lesson = await store.get_lesson(args["lesson_id"])
    if lesson:
        return [TextContent(type="text", text=json.dumps(lesson.to_dict(), ensure_ascii=False))]
    return [TextContent(type="text", text=json.dumps({"error": "Lesson not found"}))]


async def _handle_get_skill(args: Dict) -> List[TextContent]:
    skill = await store.get_skill(args["skill_id"])
    if skill:
        return [TextContent(type="text", text=json.dumps(skill.to_dict(), ensure_ascii=False))]
    return [TextContent(type="text", text=json.dumps({"error": "Skill not found"}))]


async def _handle_list_lessons(args: Dict) -> List[TextContent]:
    status = None
    if args.get("status"):
        status = LessonStatus(args["status"])
    lessons = await store.list_lessons(status=status, cluster_id=args.get("cluster_id"), limit=args.get("limit", 20))
    return [TextContent(type="text", text=json.dumps([l.to_dict() for l in lessons], ensure_ascii=False))]


async def _handle_list_skills(args: Dict) -> List[TextContent]:
    status = None
    if args.get("status"):
        status = SkillStatus(args["status"])
    skills = await store.list_skills(status=status, limit=args.get("limit", 20))
    return [TextContent(type="text", text=json.dumps([s.to_dict() for s in skills], ensure_ascii=False))]


async def main():
    """Initialize and run MCP server."""
    await store.initialize()
    # Rebuild continual index on startup
    await querier.continual_querier.rebuild_index()
    
    # Run server over stdio
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
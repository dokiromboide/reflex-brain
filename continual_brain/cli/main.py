"""
CLI Entry Points - reflex-brain command line interface.
"""
from __future__ import annotations
import os
import asyncio
import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table

from continual_brain.core.store import SQLiteStore
from continual_brain.core.refinement import RefinementEngine
from continual_brain.query.hybrid_querier import HybridQuerier
from continual_brain.daemon.processor import run_daemon

app = typer.Typer(help="Reflex Brain - Continual learning brain for AI agents")
console = Console()


@app.command()
def daemon(
    poll_interval: float = typer.Option(3.0, "--poll-interval", "-p", help="Poll interval in seconds"),
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Batch size for processing"),
    daemonize: bool = typer.Option(False, "--daemonize", "-d", help="Run as background daemon"),
):
    """Run the background processor daemon."""
    if daemonize:
        console.print("[yellow]Daemonize mode not yet implemented. Run in foreground with --daemonize flag.[/yellow]")
    console.print(f"[green]Starting Reflex Brain daemon...[/green]")
    console.print(f"Poll interval: {poll_interval}s, Batch size: {batch_size}")
    asyncio.run(run_daemon(poll_interval=poll_interval, batch_size=batch_size))


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    source_types: Optional[str] = typer.Option(None, "--source-types", "-s", help="Comma-separated source types"),
    expand_depth: int = typer.Option(1, "--expand-depth", "-e", help="Graph expansion depth"),
    min_confidence: float = typer.Option(0.0, "--min-confidence", "-c", help="Minimum confidence"),
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """Query hybrid memory."""
    console.print(f"[green]Querying: [bold]{query_text}[/bold][/green]")
    
    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())
    
    from continual_brain.query.hybrid_querier import HybridQuerier
    
    querier = HybridQuerier(
        store=store,
        brain_nodes_dir="brain/nodes",
        brain_edges_dir="brain/edges",
        brain_faiss_index="brain/brain_index.faiss",
        brain_faiss_map="brain/brain_nodes_map.pkl",
        continual_faiss_index="continual_index.faiss",
        continual_faiss_map="continual_nodes_map.pkl",
    )
    
    source_types_list = source_types.split(",") if source_types else None
    
    results = asyncio.run(querier.query(
        query_text=query_text,
        top_k=top_k,
        source_types=source_types_list,
        expand_depth=expand_depth,
        min_confidence=min_confidence,
    ))
    
    formatted = querier.format_results(results)
    console.print(formatted)


@app.command()
def list_lessons(
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    cluster_id: Optional[str] = typer.Option(None, "--cluster", help="Filter by cluster"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max results"),
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """List lessons."""
    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())
    
    from continual_brain.core.models import LessonStatus
    
    status_enum = LessonStatus(status) if status else None
    lessons = asyncio.run(store.list_lessons(status=status_enum, cluster_id=cluster_id, limit=limit))
    
    table = Table(title="Lessons")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Confidence", style="magenta")
    table.add_column("Status", style="blue")
    table.add_column("Cluster", style="dim")
    
    for lesson in lessons:
        table.add_row(
            lesson.id[:20] + "...",
            lesson.title[:50],
            str(lesson.version),
            f"{lesson.confidence:.2f}",
            lesson.status.value,
            lesson.cluster_id or "N/A",
        )
    
    console.print(table)


@app.command()
def list_skills(
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max results"),
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """List skills."""
    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())
    
    from continual_brain.core.models import SkillStatus
    
    status_enum = SkillStatus(status) if status else None
    skills = asyncio.run(store.list_skills(status=status_enum, limit=limit))
    
    table = Table(title="Skills")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Confidence", style="magenta")
    table.add_column("Status", style="blue")
    
    for skill in skills:
        table.add_row(
            skill.id[:20] + "...",
            skill.name[:50],
            str(skill.version),
            f"{skill.confidence:.2f}",
            skill.status.value,
        )
    
    console.print(table)


@app.command()
def rebuild_index(
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """Rebuild FAISS index from database."""
    console.print("[green]Rebuilding FAISS index...[/green]")
    
    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())
    
    from continual_brain.query.continual_querier import ContinualQuerier, ContinualFAISSManager
    
    faiss_mgr = ContinualFAISSManager()
    querier = ContinualQuerier(store, faiss_mgr)
    
    asyncio.run(querier.rebuild_index())
    
    console.print("[green]Index rebuilt successfully![/green]")


@app.command()
def verify(
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """Run system verification checks."""
    console.print("[green]Running system verification...[/green]")
    
    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())
    
    # Test store operations
    from continual_brain.core.models import Lesson, Skill, Memory, LessonStatus, SkillStatus
    from continual_brain.query.hybrid_querier import HybridQuerier
    
    async def run_tests():
        # Test 1: Create and fetch lesson
        lesson = Lesson(
            id="test_lesson_1",
            title="Test Lesson",
            content="This is a test lesson content.",
            evidence=[{"source": "test", "quote": "test", "weight": 0.8}],
            confidence=0.8,
            status=LessonStatus.ACCEPTED,
        )
        await store.upsert_lesson(lesson)
        fetched = await store.get_lesson("test_lesson_1")
        assert fetched is not None
        assert fetched.title == "Test Lesson"
        console.print("[green]✓[/green] Lesson CRUD works")
        
        # Test 2: Create and fetch skill
        skill = Skill(
            id="test_skill_1",
            name="test_skill",
            description="A test skill",
            code="def test(): pass",
            interface={"args": [], "returns": {}},
            confidence=0.7,
            status=SkillStatus.TESTED,
        )
        await store.upsert_skill(skill)
        fetched = await store.get_skill("test_skill_1")
        assert fetched is not None
        console.print("[green]✓[/green] Skill CRUD works")
        
        # Test 3: Create and fetch memory
        memory = Memory(
            id="test_mem_1",
            content="Test memory content",
            importance=0.8,
        )
        await store.upsert_memory(memory)
        fetched = await store.get_memory("test_mem_1")
        assert fetched is not None
        console.print("[green]✓[/green] Memory CRUD works")
        
        # Test 4: Hybrid query
        querier = HybridQuerier(
            store=store,
            brain_nodes_dir="brain/nodes",
            brain_edges_dir="brain/edges",
            brain_faiss_index="brain/brain_index.faiss",
            brain_faiss_map="brain/brain_nodes_map.pkl",
            continual_faiss_index="continual_index.faiss",
            continual_faiss_map="continual_nodes_map.pkl",
        )
        results = await querier.query("test", top_k=3)
        console.print(f"[green]✓[/green] Hybrid query works ({len(results)} results)")
        
        console.print("\n[bold green]All verification checks passed![/bold green]")
    
    asyncio.run(run_tests())


if __name__ == "__main__":
    app()
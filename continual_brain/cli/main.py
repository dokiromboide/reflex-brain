"""
CLI Entry Points - reflex-brain command line interface.
"""
from __future__ import annotations

import asyncio
import typer
from rich.console import Console
from rich.table import Table

from continual_brain.core.store import SQLiteStore
from continual_brain.daemon.processor import run_daemon
from continual_brain.query.hybrid_querier import HybridQuerier
from continual_brain.core.research_scheduler import (
    ResearchScheduler,
    ResearchTask,
    ResearchTrigger,
    ScheduleFrequency,
    TriggerType,
    create_scheduler,
    create_default_triggers,
)

app = typer.Typer(help="Reflex Brain - Continual learning brain for AI agents")
console = Console()

# Scheduler sub-commands
scheduler_app = typer.Typer(help="Research scheduler commands")
app.add_typer(scheduler_app, name="scheduler")


@app.command()
def daemon(
    poll_interval: float = typer.Option(3.0, "--poll-interval", "-p", help="Poll interval in seconds"),
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Batch size for processing"),
    daemonize: bool = typer.Option(False, "--daemonize", "-d", help="Run as background daemon"),
    enable_scheduler: bool = typer.Option(True, "--scheduler/--no-scheduler", help="Enable research scheduler"),
):
    """Run the background processor daemon."""
    if daemonize:
        console.print("[yellow]Daemonize mode not yet implemented. Run in foreground with --daemonize flag.[/yellow]")
    console.print("[green]Starting Reflex Brain daemon...[/green]")
    console.print(f"Poll interval: {poll_interval}s, Batch size: {batch_size}, Scheduler: {'enabled' if enable_scheduler else 'disabled'}")
    asyncio.run(run_daemon(poll_interval=poll_interval, batch_size=batch_size, enable_scheduler=enable_scheduler))


@app.command()
def research(
    topic: str = typer.Argument(..., help="Research topic"),
    max_sources: int = typer.Option(10, "--max-sources", "-s", help="Maximum sources to process"),
    create_lessons: bool = typer.Option(True, "--create-lessons/--no-lessons", help="Create lessons from research"),
    create_memories: bool = typer.Option(True, "--create-memories/--no-memories", help="Create episodic memories"),
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """Automated web research: search, extract, synthesize and store knowledge."""
    console.print(f"[green]Researching: [bold]{topic}[/bold][/green]")

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    from continual_brain.core.web_researcher import research_topic

    result = asyncio.run(research_topic(
        store=store,
        topic=topic,
        max_sources=max_sources,
        create_lessons=create_lessons,
        create_memories=create_memories
    ))

    console.print(f"[green]Research complete![/green]")
    console.print(f"  Sources found: {result.get('sources_found', 0)}")
    console.print(f"  Sources extracted: {result.get('sources_extracted', 0)}")
    console.print(f"  Lessons created: {result.get('lessons_created', 0)}")
    console.print(f"  Memories created: {result.get('memories_created', 0)}")

    if result.get('source_urls'):
        console.print("\n[dim]Sources:[/dim]")
        for url in result.get('source_urls', [])[:5]:
            console.print(f"  - {url}")
        if len(result.get('source_urls', [])) > 5:
            console.print(f"  ... and {len(result['source_urls']) - 5} more")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    source_types: str | None = typer.Option(None, "--source-types", "-s", help="Comma-separated source types"),
    expand_depth: int = typer.Option(1, "--expand-depth", "-e", help="Graph expansion depth"),
    min_confidence: float = typer.Option(0.0, "--min-confidence", "-c", help="Minimum confidence"),
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """Query hybrid memory."""
    console.print(f"[green]Querying: [bold]{query_text}[/bold][/green]")

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

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
    status: str | None = typer.Option(None, "--status", help="Filter by status"),
    cluster_id: str | None = typer.Option(None, "--cluster", help="Filter by cluster"),
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
    status: str | None = typer.Option(None, "--status", help="Filter by status"),
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

    from continual_brain.query.continual_querier import ContinualFAISSManager, ContinualQuerier

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
    from continual_brain.core.models import Lesson, LessonStatus, Memory, Skill, SkillStatus

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


# ============ SCHEDULER COMMANDS ============

@scheduler_app.command("add-task")
def scheduler_add_task(
    name: str = typer.Argument(..., help="Task name"),
    topic: str = typer.Argument(..., help="Research topic"),
    frequency: str = typer.Option("daily", "--frequency", "-f", help="Schedule frequency: hourly, daily, weekly, monthly, custom"),
    cron_expression: str | None = typer.Option(None, "--cron", help="Custom cron expression (if frequency=custom)"),
    max_sources: int = typer.Option(10, "--max-sources", "-s", help="Maximum sources per run"),
    create_lessons: bool = typer.Option(True, "--create-lessons/--no-lessons", help="Create lessons from research"),
    create_memories: bool = typer.Option(True, "--create-memories/--no-memories", help="Create episodic memories"),
    enabled: bool = typer.Option(True, "--enabled/--disabled", help="Enable or disable task"),
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """Add a scheduled research task."""
    console.print(f"[green]Adding scheduled task: [bold]{name}[/bold][/green]")

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    from continual_brain.core.research_scheduler import (
        ResearchScheduler, ResearchTask, ScheduleFrequency, create_scheduler
    )
    from continual_brain.query.hybrid_querier import HybridQuerier
    from continual_brain.core.web_researcher import WebResearcher

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    querier = HybridQuerier(
        store=store,
        brain_nodes_dir="brain/nodes",
        brain_edges_dir="brain/edges",
        brain_faiss_index="brain/brain_index.faiss",
        brain_faiss_map="brain/brain_nodes_map.pkl",
        continual_faiss_index="continual_index.faiss",
        continual_faiss_map="continual_nodes_map.pkl",
    )

    scheduler = create_scheduler(store, querier)
    task = ResearchTask(
        name=name,
        topic=topic,
        frequency=ScheduleFrequency(frequency),
        cron_expression=cron_expression,
        max_sources=max_sources,
        create_lessons=create_lessons,
        create_memories=create_memories,
        enabled=enabled,
    )
    scheduler.add_task(task)

    console.print(f"[green]Task created![/green]")
    console.print(f"  Task ID: {task.id}")
    console.print(f"  Next run: {task.next_run}")
    console.print(f"  Frequency: {task.frequency.value}")


@scheduler_app.command("remove-task")
def scheduler_remove_task(
    task_id: str = typer.Argument(..., help="Task ID to remove"),
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """Remove a scheduled research task."""
    console.print(f"[yellow]Removing task: [bold]{task_id}[/bold][/yellow]")

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    from continual_brain.core.research_scheduler import create_scheduler
    from continual_brain.query.hybrid_querier import HybridQuerier

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    querier = HybridQuerier(
        store=store,
        brain_nodes_dir="brain/nodes",
        brain_edges_dir="brain/edges",
        brain_faiss_index="brain/brain_index.faiss",
        brain_faiss_map="brain/brain_nodes_map.pkl",
        continual_faiss_index="continual_index.faiss",
        continual_faiss_map="continual_nodes_map.pkl",
    )

    scheduler = create_scheduler(store, querier)
    success = scheduler.remove_task(task_id)

    if success:
        console.print(f"[green]Task removed successfully![/green]")
    else:
        console.print(f"[red]Task not found: {task_id}[/red]")


@scheduler_app.command("list")
def scheduler_list_tasks(
    enabled_only: bool = typer.Option(False, "--enabled-only", "-e", help="Show only enabled tasks"),
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """List scheduled research tasks."""
    console.print("[green]Scheduled tasks:[/green]")

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    from continual_brain.core.research_scheduler import create_scheduler
    from continual_brain.query.hybrid_querier import HybridQuerier

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    querier = HybridQuerier(
        store=store,
        brain_nodes_dir="brain/nodes",
        brain_edges_dir="brain/edges",
        brain_faiss_index="brain/brain_index.faiss",
        brain_faiss_map="brain/brain_nodes_map.pkl",
        continual_faiss_index="continual_index.faiss",
        continual_faiss_map="continual_nodes_map.pkl",
    )

    scheduler = create_scheduler(store, querier)
    tasks = scheduler.list_tasks(enabled_only=enabled_only)

    if not tasks:
        console.print("[dim]No tasks found[/dim]")
        return

    table = Table(title="Scheduled Research Tasks")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Topic", style="yellow")
    table.add_column("Frequency", style="magenta")
    table.add_column("Next Run", style="blue")
    table.add_column("Enabled", style="green")
    table.add_column("Runs", style="yellow")

    for task in tasks:
        table.add_row(
            task.id[:20] + "...",
            task.name[:40],
            task.topic[:40],
            task.frequency.value,
            task.next_run or "N/A",
            "✓" if task.enabled else "✗",
            str(task.run_count),
        )

    console.print(table)


@scheduler_app.command("add-trigger")
def scheduler_add_trigger(
    name: str = typer.Argument(..., help="Trigger name"),
    topic_pattern: str = typer.Argument(..., help="Topic pattern (e.g., 'DIAN*', 'agente*')"),
    min_coverage: float = typer.Option(0.3, "--min-coverage", "-c", help="Minimum coverage score (0-1)"),
    min_results: int = typer.Option(3, "--min-results", "-r", help="Minimum results threshold"),
    cooldown: int = typer.Option(24, "--cooldown", "-c", help="Cooldown hours between triggers"),
    enabled: bool = typer.Option(True, "--enabled/--disabled", help="Enable or disable trigger"),
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """Add a low-coverage trigger for automatic research."""
    console.print(f"[green]Adding trigger: [bold]{name}[/bold][/green]")

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    from continual_brain.core.research_scheduler import (
        ResearchScheduler, ResearchTrigger, TriggerType, create_scheduler
    )
    from continual_brain.query.hybrid_querier import HybridQuerier
    from continual_brain.core.web_researcher import WebResearcher

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    querier = HybridQuerier(
        store=store,
        brain_nodes_dir="brain/nodes",
        brain_edges_dir="brain/edges",
        brain_faiss_index="brain/brain_index.faiss",
        brain_faiss_map="brain/brain_nodes_map.pkl",
        continual_faiss_index="continual_index.faiss",
        continual_faiss_map="continual_nodes_map.pkl",
    )

    scheduler = create_scheduler(store, querier)
    trigger = ResearchTrigger(
        name=name,
        topic_pattern=topic_pattern,
        min_coverage_threshold=min_coverage,
        min_results_threshold=min_results,
        cooldown_hours=cooldown,
        enabled=enabled,
    )
    scheduler.add_trigger(trigger)

    console.print(f"[green]Trigger created![/green]")
    console.print(f"  Trigger ID: {trigger.id}")
    console.print(f"  Pattern: {trigger.topic_pattern}")
    console.print(f"  Min coverage: {trigger.min_coverage_threshold}")
    console.print(f"  Min results: {trigger.min_results_threshold}")
    console.print(f"  Cooldown: {trigger.cooldown_hours}h")


@scheduler_app.command("check-coverage")
def scheduler_check_coverage(
    topic: str = typer.Argument(..., help="Topic to check"),
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """Check knowledge coverage for a topic."""
    console.print(f"[green]Checking coverage for: [bold]{topic}[/bold][/green]")

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    from continual_brain.core.research_scheduler import create_scheduler
    from continual_brain.query.hybrid_querier import HybridQuerier

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    querier = HybridQuerier(
        store=store,
        brain_nodes_dir="brain/nodes",
        brain_edges_dir="brain/edges",
        brain_faiss_index="brain/brain_index.faiss",
        brain_faiss_map="brain/brain_nodes_map.pkl",
        continual_faiss_index="continual_index.faiss",
        continual_faiss_map="continual_nodes_map.pkl",
    )

    from continual_brain.core.research_scheduler import create_scheduler
    scheduler = create_scheduler(store, querier)

    coverage = asyncio.run(scheduler.check_coverage(topic))

    console.print(f"[green]Coverage for: [bold]{topic}[/bold][/green]")
    console.print(f"  Covered: {'✓' if coverage['covered'] else '✗'}")
    console.print(f"  Score: {coverage['score']:.3f}")
    console.print(f"  Max score: {coverage['max_score']:.3f}")
    console.print(f"  Results: {coverage['result_count']}")
    console.print(f"  Source types: {coverage['source_types']}")
    if coverage.get('top_result'):
        console.print(f"  Top result: {coverage['top_result']['title']} ({coverage['top_result']['score']:.3f})")


@scheduler_app.command("stats")
def scheduler_stats(
    db_path: str = typer.Option("continual.db", "--db", help="Database path"),
):
    """Get scheduler statistics."""
    console.print("[green]Scheduler Statistics:[/green]")

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    from continual_brain.core.research_scheduler import create_scheduler
    from continual_brain.query.hybrid_querier import HybridQuerier

    store = SQLiteStore(db_path)
    asyncio.run(store.initialize())

    querier = HybridQuerier(
        store=store,
        brain_nodes_dir="brain/nodes",
        brain_edges_dir="brain/edges",
        brain_faiss_index="brain/brain_index.faiss",
        brain_faiss_map="brain/brain_nodes_map.pkl",
        continual_faiss_index="continual_index.faiss",
        continual_faiss_map="continual_nodes_map.pkl",
    )

    from continual_brain.core.research_scheduler import create_scheduler
    scheduler = create_scheduler(store, querier)

    stats = scheduler.get_stats()

    table = Table(title="Scheduler Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


if __name__ == "__main__":
    app()
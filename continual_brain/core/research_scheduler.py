"""
Research Scheduler - Automated scheduled research tasks.
Monitors topics, triggers research on low coverage, runs periodic investigations.
"""
from __future__ import annotations
import asyncio
import json
import os
import uuid
import logging
import tempfile
from filelock import FileLock, Timeout
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from pathlib import Path

from continual_brain.core.store import SQLiteStore
from continual_brain.core.models import Lesson, Memory, LessonStatus, MemoryType
from continual_brain.core.web_researcher import WebResearcher, research_topic
from continual_brain.query.hybrid_querier import HybridQuerier
from continual_brain.query.continual_querier import ContinualQuerier, ContinualFAISSManager


logger = logging.getLogger(__name__)

def utcnow():
    return datetime.now(timezone.utc)

def parse_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed

class ScheduleFrequency(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class TriggerType(str, Enum):
    SCHEDULED = "scheduled"
    LOW_COVERAGE = "low_coverage"
    MANUAL = "manual"
    TOPIC_WATCH = "topic_watch"


@dataclass
class ResearchTask:
    """A scheduled research task."""
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    name: str = ""
    topic: str = ""
    frequency: ScheduleFrequency = ScheduleFrequency.DAILY
    cron_expression: Optional[str] = None  # For custom schedules
    max_sources: int = 10
    create_lessons: bool = True
    create_memories: bool = True
    enabled: bool = True
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    run_count: int = 0
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: utcnow().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert frequency string to enum if needed."""
        if isinstance(self.frequency, str):
            self.frequency = ScheduleFrequency(self.frequency)
        if not 1 <= self.max_sources <= 50:
            raise ValueError("max_sources must be between 1 and 50")
        if self.frequency == ScheduleFrequency.CUSTOM:
            from croniter import croniter
            if not self.cron_expression or not croniter.is_valid(self.cron_expression):
                raise ValueError("Custom frequency requires a valid cron_expression")


@dataclass
class ResearchTrigger:
    """Trigger for automatic research."""
    id: str = field(default_factory=lambda: f"trigger_{uuid.uuid4().hex[:12]}")
    name: str = ""
    trigger_type: TriggerType = TriggerType.LOW_COVERAGE
    topic_pattern: str = ""  # e.g., "DIAN*", "agente*"
    min_coverage_threshold: float = 0.3  # Minimum hybrid query score
    min_results_threshold: int = 3  # Minimum results from hybrid query
    cooldown_hours: int = 24  # Minimum time between triggers
    enabled: bool = True
    last_triggered: Optional[str] = None
    trigger_count: int = 0
    created_at: str = field(default_factory=lambda: utcnow().isoformat())

    def __post_init__(self):
        """Convert trigger_type string to enum if needed."""
        if isinstance(self.trigger_type, str):
            self.trigger_type = TriggerType(self.trigger_type)


@dataclass
class ResearchJob:
    """A research job execution record."""
    id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:12]}")
    task_id: Optional[str] = None
    trigger_id: Optional[str] = None
    topic: str = ""
    trigger_type: TriggerType = TriggerType.MANUAL
    status: str = "pending"  # pending, running, completed, failed
    started_at: str = field(default_factory=lambda: utcnow().isoformat())
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self):
        """Convert trigger_type string to enum if needed."""
        if isinstance(self.trigger_type, str):
            self.trigger_type = TriggerType(self.trigger_type)


class ResearchScheduler:
    """Orchestrates scheduled and triggered research tasks."""
    
    def __init__(
        self,
        store: SQLiteStore,
        hybrid_querier: HybridQuerier,
        web_researcher: WebResearcher,
        check_interval_seconds: int = 300,  # 5 minutes
        state_dir: str = "scheduler_state"
    ):
        self.store = store
        self.hybrid_querier = hybrid_querier
        self.web_researcher = web_researcher
        self.check_interval = check_interval_seconds
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.tasks: Dict[str, ResearchTask] = {}
        self.triggers: Dict[str, ResearchTrigger] = {}
        self.jobs: List[ResearchJob] = []
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        
        self._workers = set()
        self._state_lock = FileLock(str(self.state_dir / "state.lock"), timeout=5)
        self._runner_lock = FileLock(str(self.state_dir / "runner.lock"))
        self._baseline = {"tasks": {}, "triggers": {}, "jobs": {}}
        # Load persisted state
        self._load_state()
    
    def _read_state(self):
        path = self.state_dir / "state.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        state = {}
        for kind in ("tasks", "triggers", "jobs"):
            old = self.state_dir / f"{kind}.json"
            state[kind] = json.loads(old.read_text(encoding="utf-8")) if old.exists() else []
        return state

    def _load_state(self):
        with self._state_lock:
            state = self._read_state()
        self.tasks = {v["id"]: ResearchTask(**v) for v in state["tasks"]}
        self.triggers = {v["id"]: ResearchTrigger(**v) for v in state["triggers"]}
        self.jobs = [ResearchJob(**v) for v in state["jobs"]]
        self._baseline = {kind: {v["id"]: v for v in state[kind]} for kind in state}

    def _save_state(self):
        # Merge only changed records while holding a cross-process lock. A second
        # MCP client can edit schedules without overwriting the runner's jobs.
        current = {"tasks": [t.__dict__.copy() for t in self.tasks.values()],
                   "triggers": [t.__dict__.copy() for t in self.triggers.values()],
                   "jobs": [t.__dict__.copy() for t in self.jobs]}
        with self._state_lock:
            disk = self._read_state()
            for kind, values in current.items():
                merged = {v["id"]: v for v in disk[kind]}
                local = {v["id"]: v for v in values}
                previous = self._baseline[kind]
                for key in previous.keys() - local.keys():
                    if merged.get(key) == previous[key]:
                        merged.pop(key, None)
                for key, value in local.items():
                    if value != previous.get(key):
                        if key in previous and merged.get(key) not in (previous[key], value):
                            raise ValueError("Scheduler record changed in another process; retry")
                        merged[key] = value
                disk[kind] = list(merged.values())
            disk["jobs"] = disk["jobs"][-1000:]
            fd, filename = tempfile.mkstemp(dir=self.state_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(disk, handle, ensure_ascii=False, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(filename, self.state_dir / "state.json")
            finally:
                if os.path.exists(filename):
                    os.unlink(filename)
            self._baseline = {kind: {v["id"]: v for v in values} for kind, values in current.items()}

    # ============ Task Management ============
    
    def add_task(self, task: ResearchTask) -> ResearchTask:
        """Add a scheduled research task."""
        # Calculate next run
        task.next_run = self._calculate_next_run(task)
        self.tasks[task.id] = task
        self._save_state()
        return task
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a scheduled task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._save_state()
            return True
        return False
    
    def get_task(self, task_id: str) -> Optional[ResearchTask]:
        return self.tasks.get(task_id)
    
    def list_tasks(self, enabled_only: bool = False) -> List[ResearchTask]:
        tasks = list(self.tasks.values())
        if enabled_only:
            tasks = [t for t in tasks if t.enabled]
        return sorted(tasks, key=lambda t: t.next_run or "")
    
    def update_task(self, task_id: str, **kwargs) -> Optional[ResearchTask]:
        """Update task properties."""
        if task_id not in self.tasks:
            return None
        task = self.tasks[task_id]
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)
        task.__post_init__()
        task.updated_at = utcnow().isoformat()
        if "frequency" in kwargs or "cron_expression" in kwargs:
            task.next_run = self._calculate_next_run(task)
        self._save_state()
        return task
    
    # ============ Trigger Management ============
    
    def add_trigger(self, trigger: ResearchTrigger) -> ResearchTrigger:
        """Add a low-coverage trigger."""
        self.triggers[trigger.id] = trigger
        self._save_state()
        return trigger
    
    def remove_trigger(self, trigger_id: str) -> bool:
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            self._save_state()
            return True
        return False
    
    def get_trigger(self, trigger_id: str) -> Optional[ResearchTrigger]:
        return self.triggers.get(trigger_id)
    
    def list_triggers(self, enabled_only: bool = False) -> List[ResearchTrigger]:
        triggers = list(self.triggers.values())
        if enabled_only:
            triggers = [t for t in triggers if t.enabled]
        return triggers
    
    # ============ Coverage Checking ============
    
    async def check_coverage(self, topic: str) -> Dict[str, Any]:
        """Check knowledge coverage for a topic using hybrid query."""
        results = await self.hybrid_querier.query(
            query_text=topic,
            top_k=10,
            source_types=["lesson", "skill", "memory", "knowledge", "conversation"]
        )
        
        if not results:
            return {
                "topic": topic,
                "covered": False,
                "score": 0.0,
                "result_count": 0,
                "max_score": 0.0,
                "source_types": {}
            }
        
        max_score = max(r.score for r in results)
        avg_score = sum(r.score for r in results) / len(results)
        source_types = {}
        for r in results:
            st = r.source_type
            source_types[st] = source_types.get(st, 0) + 1
        
        return {
            "topic": topic,
            "covered": max_score >= 0.3,  # Default threshold
            "score": avg_score,
            "max_score": max_score,
            "result_count": len(results),
            "source_types": source_types,
            "top_result": {
                "title": results[0].title,
                "score": results[0].score,
                "source_type": results[0].source_type
            } if results else None
        }
    
    async def check_triggers(self) -> List[ResearchJob]:
        """Check all triggers and create jobs for triggered ones."""
        triggered_jobs = []
        
        for trigger in list(self.triggers.values()):
            if not trigger.enabled:
                continue
            
            # Check cooldown
            if trigger.last_triggered:
                last = parse_time(trigger.last_triggered)
                if utcnow() - last < timedelta(hours=trigger.cooldown_hours):
                    continue
            
            # Check topic pattern
            topics_to_check = self._expand_topic_pattern(trigger.topic_pattern)
            
            for topic in topics_to_check:
                coverage = await self.check_coverage(topic)
                
                # Check if coverage is below threshold
                if (coverage["max_score"] < trigger.min_coverage_threshold or 
                    coverage["result_count"] < trigger.min_results_threshold):
                    
                    # Create job
                    job = ResearchJob(
                        trigger_id=trigger.id,
                        topic=topic,
                        trigger_type=TriggerType.LOW_COVERAGE,
                        status="pending"
                    )
                    self.jobs.append(job)
                    triggered_jobs.append(job)
                    
                    # Update trigger
                    trigger.last_triggered = utcnow().isoformat()
                    trigger.trigger_count += 1
        
        if triggered_jobs:
            self._save_state()
        
        return triggered_jobs
    
    def _expand_topic_pattern(self, pattern: str) -> List[str]:
        """Expand topic pattern (e.g., 'DIAN*' -> ['DIAN facturación', 'DIAN IVA', ...])."""
        if not pattern.endswith("*"):
            return [pattern]
        
        prefix = pattern[:-1]
        # Search existing lessons for matching topics
        topics = set()
        # This would ideally search the knowledge base
        # For now, return the prefix as a topic to research
        return [prefix]
    
    # ============ Scheduler Loop ============
    
    async def start(self):
        """Start the scheduler loop."""
        if self.running:
            return
        
        try:
            self._runner_lock.acquire(timeout=0)
        except Timeout:
            logger.info("Another MCP process owns the research scheduler")
            return
        self._load_state()
        for job in self.jobs:
            if job.status == "running":
                job.status = "failed"
                job.error = "Previous process interrupted; manual retry required"
                job.completed_at = utcnow().isoformat()
        self._save_state()
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop(self):
        """Stop the scheduler loop."""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
    
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        if self._runner_lock.is_locked:
            self._runner_lock.release()

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                if not self._workers:
                    self._load_state()
                await self._process_due_tasks()
                await self.check_triggers()
                await self._process_pending_jobs()
            except Exception as e:
                logger.exception("Scheduler error")
            
            await asyncio.sleep(self.check_interval)
    
    async def _process_due_tasks(self):
        """Execute tasks that are due."""
        now = utcnow()
        
        for task in list(self.tasks.values()):
            if not task.enabled or not task.next_run:
                continue
            
            next_run = parse_time(task.next_run)
            if now >= next_run:
                # Execute task
                await self._execute_task(task)
                
                # Update next run
                task.last_run = utcnow().isoformat()
                task.run_count += 1
                task.next_run = self._calculate_next_run(task)
                self._save_state()
    
    async def _execute_task(self, task: ResearchTask):
        """Execute a research task."""
        job = ResearchJob(
            task_id=task.id,
            topic=task.topic,
            trigger_type=TriggerType.SCHEDULED,
            status="running"
        )
        self.jobs.append(job)
        
        try:
            result = await self.web_researcher.research_topic(
                topic=task.topic,
                max_sources=task.max_sources,
                create_lessons=task.create_lessons,
                create_memories=task.create_memories
            )
            
            if result.get("error"):
                raise ValueError(result["error"])
            job.status = "completed"
            job.completed_at = utcnow().isoformat()
            job.result = result
        except asyncio.CancelledError:
            job.status = "failed"
            job.error = "Research interrupted by shutdown"
            job.completed_at = utcnow().isoformat()
            raise
        except Exception as e:
            job.status = "failed"
            job.completed_at = utcnow().isoformat()
            job.error = str(e)
        finally:
            self._save_state()
    
    async def _process_pending_jobs(self):
        """Process pending jobs (limit concurrent)."""
        running_jobs = [j for j in self.jobs if j.status == "running"]
        if len(running_jobs) >= 3:  # Max concurrent
            return
        
        pending_jobs = [j for j in self.jobs if j.status == "pending"]
        for job in pending_jobs[:3 - len(running_jobs)]:
            job.status = "running"
            worker = asyncio.create_task(self._execute_job(job))
            self._workers.add(worker)
            worker.add_done_callback(self._workers.discard)
        self._save_state()
    
    async def _execute_job(self, job: ResearchJob):
        """Execute a research job (manual or triggered)."""
        try:
            result = await self.web_researcher.research_topic(
                topic=job.topic,
                max_sources=10,
                create_lessons=True,
                create_memories=True
            )
            
            if result.get("error"):
                raise ValueError(result["error"])
            job.status = "completed"
            job.completed_at = utcnow().isoformat()
            job.result = result
        except asyncio.CancelledError:
            job.status = "failed"
            job.error = "Research interrupted by shutdown"
            job.completed_at = utcnow().isoformat()
            raise
        except Exception as e:
            job.status = "failed"
            job.completed_at = utcnow().isoformat()
            job.error = str(e)
        finally:
            self._save_state()
    
    def _calculate_next_run(self, task: ResearchTask) -> Optional[str]:
        """Calculate next run time based on frequency."""
        now = utcnow()
        
        if task.frequency == ScheduleFrequency.HOURLY:
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif task.frequency == ScheduleFrequency.DAILY:
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
        elif task.frequency == ScheduleFrequency.WEEKLY:
            days_ahead = 6 - now.weekday()  # Next Sunday
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
        elif task.frequency == ScheduleFrequency.MONTHLY:
            if now.month == 12:
                next_run = now.replace(year=now.year + 1, month=1, day=1, hour=2, minute=0, second=0, microsecond=0)
            else:
                next_run = now.replace(month=now.month + 1, day=1, hour=2, minute=0, second=0, microsecond=0)
        elif task.frequency == ScheduleFrequency.CUSTOM:
            from croniter import croniter
            if not task.cron_expression or not croniter.is_valid(task.cron_expression):
                raise ValueError("Custom frequency requires a valid cron_expression")
            next_run = croniter(task.cron_expression, now).get_next(datetime)
        else:
            raise ValueError("Unsupported frequency")
        if next_run <= now:
            next_run += timedelta(days=7 if task.frequency == ScheduleFrequency.WEEKLY else 1)
        return next_run.isoformat()

    # ============ Manual Research ============
    
    async def research_now(self, topic: str, max_sources: int = 10) -> Dict[str, Any]:
        """Manually trigger research on a topic."""
        return await research_topic(
            store=self.store,
            topic=topic,
            max_sources=max_sources,
            create_lessons=True,
            create_memories=True
        )
    
    def get_job_history(self, limit: int = 50) -> List[ResearchJob]:
        return sorted(self.jobs, key=lambda j: j.started_at, reverse=True)[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "tasks_total": len(self.tasks),
            "tasks_enabled": sum(1 for t in self.tasks.values() if t.enabled),
            "triggers_total": len(self.triggers),
            "triggers_enabled": sum(1 for t in self.triggers.values() if t.enabled),
            "jobs_total": len(self.jobs),
            "jobs_pending": sum(1 for j in self.jobs if j.status == "pending"),
            "jobs_running": sum(1 for j in self.jobs if j.status == "running"),
            "jobs_completed": sum(1 for j in self.jobs if j.status == "completed"),
            "jobs_failed": sum(1 for j in self.jobs if j.status == "failed"),
            "scheduler_running": self.running,
        }


# Convenience function for creating default scheduler
def create_scheduler(
    store: SQLiteStore,
    hybrid_querier: HybridQuerier,
    check_interval_seconds: int = 300
) -> ResearchScheduler:
    """Create a ResearchScheduler with default components."""
    web_researcher = WebResearcher(store)
    return ResearchScheduler(
        store=store,
        hybrid_querier=hybrid_querier,
        web_researcher=web_researcher,
        check_interval_seconds=check_interval_seconds
    )


# Default triggers for common topics
def create_default_triggers(scheduler: ResearchScheduler) -> List[ResearchTrigger]:
    """Create default triggers for common Colombian business topics."""
    defaults = [
        ResearchTrigger(
            name="DIAN Monitoring",
            trigger_type=TriggerType.LOW_COVERAGE,
            topic_pattern="DIAN*",
            min_coverage_threshold=0.25,
            min_results_threshold=2,
            cooldown_hours=12,
        ),
        ResearchTrigger(
            name="Agent Architecture Monitoring",
            trigger_type=TriggerType.LOW_COVERAGE,
            topic_pattern="agente*",
            min_coverage_threshold=0.2,
            min_results_threshold=2,
            cooldown_hours=24,
        ),
        ResearchTrigger(
            name="Novel Writing Monitoring",
            trigger_type=TriggerType.LOW_COVERAGE,
            topic_pattern="novela*",
            min_coverage_threshold=0.15,
            min_results_threshold=1,
            cooldown_hours=48,
        ),
        ResearchTrigger(
            name="Colombian Business Compliance",
            trigger_type=TriggerType.LOW_COVERAGE,
            topic_pattern="facturación*",
            min_coverage_threshold=0.25,
            min_results_threshold=2,
            cooldown_hours=12,
        ),
    ]
    
    for trigger in defaults:
        trigger.id = "default_" + trigger.name.lower().replace(" ", "_")
        if trigger.id not in scheduler.triggers:
            scheduler.add_trigger(trigger)
    
    return defaults

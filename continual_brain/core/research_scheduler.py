"""
Research Scheduler - Automated scheduled research tasks.
Monitors topics, triggers research on low coverage, runs periodic investigations.
"""
from __future__ import annotations
import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from pathlib import Path

from continual_brain.core.store import SQLiteStore
from continual_brain.core.models import Lesson, Memory, LessonStatus, MemoryType
from continual_brain.core.web_researcher import WebResearcher, research_topic
from continual_brain.query.hybrid_querier import HybridQuerier
from continual_brain.query.continual_querier import ContinualQuerier, ContinualFAISSManager


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
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert frequency string to enum if needed."""
        if isinstance(self.frequency, str):
            self.frequency = ScheduleFrequency(self.frequency)


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
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

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
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
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
        
        # Load persisted state
        self._load_state()
    
    def _load_state(self):
        """Load persisted tasks, triggers, and jobs."""
        # Tasks
        tasks_file = self.state_dir / "tasks.json"
        if tasks_file.exists():
            with open(tasks_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for t in data:
                    task = ResearchTask(**t)
                    self.tasks[task.id] = task
        
        # Triggers
        triggers_file = self.state_dir / "triggers.json"
        if triggers_file.exists():
            with open(triggers_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for t in data:
                    trigger = ResearchTrigger(**t)
                    self.triggers[trigger.id] = trigger
        
        # Jobs
        jobs_file = self.state_dir / "jobs.json"
        if jobs_file.exists():
            with open(jobs_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.jobs = [ResearchJob(**j) for j in data]
    
    def _save_state(self):
        """Persist state to disk."""
        # Tasks
        with open(self.state_dir / "tasks.json", 'w', encoding='utf-8') as f:
            json.dump([t.__dict__ for t in self.tasks.values()], f, ensure_ascii=False, indent=2)
        
        # Triggers
        with open(self.state_dir / "triggers.json", 'w', encoding='utf-8') as f:
            json.dump([t.__dict__ for t in self.triggers.values()], f, ensure_ascii=False, indent=2)
        
        # Jobs (keep last 1000)
        jobs_to_save = self.jobs[-1000:]
        with open(self.state_dir / "jobs.json", 'w', encoding='utf-8') as f:
            json.dump([j.__dict__ for j in jobs_to_save], f, ensure_ascii=False, indent=2)
    
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
        task.updated_at = datetime.utcnow().isoformat() + "Z"
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
        
        for trigger in self.triggers.values():
            if not trigger.enabled:
                continue
            
            # Check cooldown
            if trigger.last_triggered:
                last = datetime.fromisoformat(trigger.last_triggered.replace('Z', '+00:00'))
                if datetime.utcnow() - last < timedelta(hours=trigger.cooldown_hours):
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
                    trigger.last_triggered = datetime.utcnow().isoformat() + "Z"
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
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                await self._process_due_tasks()
                await self.check_triggers()
                await self._process_pending_jobs()
            except Exception as e:
                print(f"Scheduler error: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def _process_due_tasks(self):
        """Execute tasks that are due."""
        now = datetime.utcnow()
        
        for task in self.tasks.values():
            if not task.enabled or not task.next_run:
                continue
            
            next_run = datetime.fromisoformat(task.next_run.replace('Z', '+00:00'))
            if now >= next_run:
                # Execute task
                await self._execute_task(task)
                
                # Update next run
                task.last_run = datetime.utcnow().isoformat() + "Z"
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
            result = await research_topic(
                store=self.store,
                topic=task.topic,
                max_sources=task.max_sources,
                create_lessons=task.create_lessons,
                create_memories=task.create_memories
            )
            
            job.status = "completed"
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            job.result = result
        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.utcnow().isoformat() + "Z"
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
            asyncio.create_task(self._execute_job(job))
    
    async def _execute_job(self, job: ResearchJob):
        """Execute a research job (manual or triggered)."""
        try:
            result = await research_topic(
                store=self.store,
                topic=job.topic,
                max_sources=10,
                create_lessons=True,
                create_memories=True
            )
            
            job.status = "completed"
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            job.result = result
        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.utcnow().isoformat() + "Z"
            job.error = str(e)
        finally:
            self._save_state()
    
    def _calculate_next_run(self, task: ResearchTask) -> Optional[str]:
        """Calculate next run time based on frequency."""
        now = datetime.utcnow()
        
        if task.frequency == ScheduleFrequency.HOURLY:
            next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        elif task.frequency == ScheduleFrequency.DAILY:
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif task.frequency == ScheduleFrequency.WEEKLY:
            days_ahead = 6 - now.weekday()  # Next Sunday
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
        elif task.frequency == ScheduleFrequency.MONTHLY:
            if now.month == 12:
                next_run = now.replace(year=now.year + 1, month=1, day=1, hour=2, minute=0, second=0, microsecond=0)
            else:
                next_run = now.replace(month=now.month + 1, day=1, hour=2, minute=0, second=0, microsecond=0)
        elif task.frequency == ScheduleFrequency.CUSTOM and task.cron_expression:
            # Use croniter for custom cron expressions
            try:
                from croniter import croniter
                cron = croniter(task.cron_expression, now)
                next_run = cron.get_next(datetime)
            except Exception as e:
                print(f"Invalid cron expression: {task.cron_expression}, error: {e}")
                next_run = now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=1)
        elif task.frequency == ScheduleFrequency.CUSTOM:
            # Custom frequency without cron expression - fall back to daily
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            return None
        
        return next_run.isoformat() + "Z"
    
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
        scheduler.add_trigger(trigger)
    
    return defaults
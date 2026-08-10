"""
Integration tests for Research Scheduler.
"""
import pytest
import asyncio
import tempfile
import os

from continual_brain.core.store import SQLiteStore
from continual_brain.core.research_scheduler import (
    ResearchScheduler,
    ResearchTask,
    ResearchTrigger,
    ScheduleFrequency,
    TriggerType,
    create_scheduler,
    create_default_triggers,
)
from continual_brain.query.hybrid_querier import HybridQuerier
from continual_brain.core.web_researcher import WebResearcher
from continual_brain.query.continual_querier import ContinualQuerier, ContinualFAISSManager


@pytest.fixture
async def temp_store():
    """Create a temporary store for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    store = SQLiteStore(db_path)
    await store.initialize()

    yield store

    await store.close()
    os.unlink(db_path)


@pytest.fixture
async def temp_querier(temp_store):
    """Create a temporary hybrid querier for testing."""
    from continual_brain.query.continual_querier import ContinualFAISSManager, ContinualQuerier
    
    faiss_mgr = ContinualFAISSManager()
    querier = ContinualQuerier(temp_store, faiss_mgr)
    await querier.rebuild_index()
    
    hybrid = HybridQuerier(
        store=temp_store,
        brain_nodes_dir="brain/nodes",
        brain_edges_dir="brain/edges",
        brain_faiss_index="brain/brain_index.faiss",
        brain_faiss_map="brain/brain_nodes_map.pkl",
        continual_faiss_index="continual_index.faiss",
        continual_faiss_map="continual_nodes_map.pkl",
    )
    
    yield hybrid


@pytest.fixture
async def temp_web_researcher(temp_store):
    """Create a temporary web researcher for testing."""
    researcher = WebResearcher(temp_store)
    yield researcher
    await researcher.close()


class TestResearchScheduler:
    """Integration tests for ResearchScheduler."""
    
    @pytest.mark.asyncio
    async def test_scheduler_creation(self, temp_store, temp_querier, temp_web_researcher):
        """Test scheduler creation and basic functionality."""
        scheduler = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=tempfile.mkdtemp()
        )
        
        assert scheduler is not None
        assert len(scheduler.tasks) == 0
        assert len(scheduler.triggers) == 0
        assert len(scheduler.jobs) == 0
        assert scheduler.running is False
    
    @pytest.mark.asyncio
    async def test_add_and_remove_task(self, temp_store, temp_querier, temp_web_researcher):
        """Test adding and removing scheduled tasks."""
        scheduler = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=tempfile.mkdtemp()
        )
        
        task = ResearchTask(
            name="Test Task",
            topic="test topic",
            frequency=ScheduleFrequency.DAILY,
            max_sources=5,
        )
        
        scheduler.add_task(task)
        assert len(scheduler.tasks) == 1
        assert task.id in scheduler.tasks
        
        # Test get task
        fetched = scheduler.get_task(task.id)
        assert fetched is not None
        assert fetched.name == "Test Task"
        
        # Test list tasks
        tasks = scheduler.list_tasks()
        assert len(tasks) == 1
        
        # Test remove task
        removed = scheduler.remove_task(task.id)
        assert removed is True
        assert len(scheduler.tasks) == 0
        
        # Test remove non-existent
        removed = scheduler.remove_task("non_existent")
        assert removed is False
    
    @pytest.mark.asyncio
    async def test_task_frequency_enum_conversion(self, temp_store, temp_querier, temp_web_researcher):
        """Test that frequency string is converted to enum."""
        scheduler = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=tempfile.mkdtemp()
        )
        
        # Test with string frequency (as loaded from JSON)
        task = ResearchTask(
            name="Test Task",
            topic="test topic",
            frequency="weekly",  # String instead of enum
        )
        
        # The __post_init__ should convert string to enum
        assert task.frequency == ScheduleFrequency.WEEKLY
    
    @pytest.mark.asyncio
    async def test_add_and_remove_trigger(self, temp_store, temp_querier, temp_web_researcher):
        """Test adding and removing triggers."""
        scheduler = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=tempfile.mkdtemp()
        )
        
        trigger = ResearchTrigger(
            name="Test Trigger",
            topic_pattern="DIAN*",
            min_coverage_threshold=0.25,
        )
        
        scheduler.add_trigger(trigger)
        assert len(scheduler.triggers) == 1
        assert trigger.id in scheduler.triggers
        
        # Test get trigger
        fetched = scheduler.get_trigger(trigger.id)
        assert fetched is not None
        assert fetched.name == "Test Trigger"
        
        # Test list triggers
        triggers = scheduler.list_triggers()
        assert len(triggers) == 1
        
        # Test remove trigger
        removed = scheduler.remove_trigger(trigger.id)
        assert removed is True
        assert len(scheduler.triggers) == 0
    
    @pytest.mark.asyncio
    async def test_trigger_enum_conversion(self, temp_store, temp_querier, temp_web_researcher):
        """Test that trigger_type string is converted to enum."""
        scheduler = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=tempfile.mkdtemp()
        )
        
        trigger = ResearchTrigger(
            name="Test Trigger",
            topic_pattern="DIAN*",
            trigger_type="manual",  # String instead of enum
        )
        
        # The __post_init__ should convert string to enum
        assert trigger.trigger_type == TriggerType.MANUAL
    
    @pytest.mark.asyncio
    async def test_check_coverage(self, temp_store, temp_querier, temp_web_researcher):
        """Test coverage checking functionality."""
        scheduler = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=tempfile.mkdtemp()
        )
        
        # Test coverage for empty topic
        coverage = await scheduler.check_coverage("nonexistent topic xyz")
        assert coverage["covered"] is False
        assert coverage["score"] == 0.0
        assert coverage["result_count"] == 0
    
    @pytest.mark.asyncio
    async def test_calculate_next_run(self, temp_store, temp_querier, temp_web_researcher):
        """Test next run calculation for different frequencies."""
        scheduler = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=tempfile.mkdtemp()
        )
        
        from datetime import datetime
        
        # Test hourly
        task = ResearchTask(
            name="Hourly Task",
            topic="test",
            frequency=ScheduleFrequency.HOURLY,
        )
        next_run = scheduler._calculate_next_run(task)
        assert next_run is not None
        
        # Test daily
        task = ResearchTask(
            name="Daily Task",
            topic="test",
            frequency=ScheduleFrequency.DAILY,
        )
        next_run = scheduler._calculate_next_run(task)
        assert next_run is not None
        
        # Test weekly
        task = ResearchTask(
            name="Weekly Task",
            topic="test",
            frequency=ScheduleFrequency.WEEKLY,
        )
        next_run = scheduler._calculate_next_run(task)
        assert next_run is not None
        
        # Test monthly
        task = ResearchTask(
            name="Monthly Task",
            topic="test",
            frequency=ScheduleFrequency.MONTHLY,
        )
        next_run = scheduler._calculate_next_run(task)
        assert next_run is not None
        
        # Test custom with cron
        task = ResearchTask(
            name="Custom Task",
            topic="test",
            frequency=ScheduleFrequency.CUSTOM,
            cron_expression="0 */6 * * *",  # Every 6 hours
        )
        next_run = scheduler._calculate_next_run(task)
        assert next_run is not None
        
        # Test custom without cron (should fall back to daily)
        task = ResearchTask(
            name="Custom No Cron",
            topic="test",
            frequency=ScheduleFrequency.CUSTOM,
            cron_expression=None,
        )
        next_run = scheduler._calculate_next_run(task)
        assert next_run is not None
    
    @pytest.mark.asyncio
    async def test_get_stats(self, temp_store, temp_querier, temp_web_researcher):
        """Test scheduler statistics."""
        scheduler = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=tempfile.mkdtemp()
        )
        
        stats = scheduler.get_stats()
        assert stats["tasks_total"] == 0
        assert stats["tasks_enabled"] == 0
        assert stats["triggers_total"] == 0
        assert stats["triggers_enabled"] == 0
        assert stats["jobs_total"] == 0
        assert stats["scheduler_running"] is False
        
        # Add a task and check stats
        task = ResearchTask(name="Test", topic="test", frequency=ScheduleFrequency.DAILY)
        scheduler.add_task(task)
        
        stats = scheduler.get_stats()
        assert stats["tasks_total"] == 1
        assert stats["tasks_enabled"] == 1
    
    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, temp_store, temp_querier, temp_web_researcher):
        """Test scheduler start/stop."""
        scheduler = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=tempfile.mkdtemp()
        )
        
        assert scheduler.running is False
        
        await scheduler.start()
        assert scheduler.running is True
        assert scheduler.scheduler_task is not None
        
        await scheduler.stop()
        assert scheduler.running is False
    
    @pytest.mark.asyncio
    async def test_create_default_triggers(self, temp_store, temp_querier, temp_web_researcher):
        """Test creating default triggers."""
        scheduler = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=tempfile.mkdtemp()
        )
        
        triggers = create_default_triggers(scheduler)
        assert len(triggers) == 4
        
        # Check trigger names
        names = [t.name for t in triggers]
        assert "DIAN Monitoring" in names
        assert "Agent Architecture Monitoring" in names
        assert "Novel Writing Monitoring" in names
        assert "Colombian Business Compliance" in names
        
        # Check they were added to scheduler
        assert len(scheduler.triggers) == 4


class TestCreateScheduler:
    """Tests for create_scheduler factory function."""
    
    @pytest.mark.asyncio
    async def test_create_scheduler(self, temp_store, temp_querier, temp_web_researcher):
        """Test create_scheduler factory."""
        scheduler = create_scheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            check_interval_seconds=10,
        )
        
        assert scheduler is not None
        assert isinstance(scheduler, ResearchScheduler)
        assert scheduler.check_interval == 10
    
    @pytest.mark.asyncio
    async def test_create_default_triggers_function(self, temp_store, temp_querier, temp_web_researcher):
        """Test create_default_triggers function."""
        import tempfile
        state_dir = tempfile.mkdtemp()
        
        scheduler = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=10,
            state_dir=state_dir
        )
        
        triggers = create_default_triggers(scheduler)
        assert len(triggers) == 4
        assert len(scheduler.triggers) == 4


class TestSchedulerPersistence:
    """Tests for scheduler state persistence."""
    
    @pytest.mark.asyncio
    async def test_persistence(self, temp_store, temp_querier, temp_web_researcher):
        """Test that tasks and triggers persist across scheduler instances."""
        state_dir = tempfile.mkdtemp()
        
        # Create first scheduler and add task/trigger
        scheduler1 = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=state_dir
        )
        
        task = ResearchTask(name="Persisted Task", topic="test", frequency=ScheduleFrequency.DAILY)
        trigger = ResearchTrigger(name="Persisted Trigger", topic_pattern="DIAN*")
        
        scheduler1.add_task(task)
        scheduler1.add_trigger(trigger)
        
        # Create second scheduler with same state_dir
        scheduler2 = ResearchScheduler(
            store=temp_store,
            hybrid_querier=temp_querier,
            web_researcher=temp_web_researcher,
            check_interval_seconds=1,
            state_dir=state_dir
        )
        
        # Should load persisted data
        assert len(scheduler2.tasks) == 1
        assert len(scheduler2.triggers) == 1
        
        loaded_task = list(scheduler2.tasks.values())[0]
        loaded_trigger = list(scheduler2.triggers.values())[0]
        
        assert loaded_task.name == "Persisted Task"
        assert loaded_trigger.name == "Persisted Trigger"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Regression tests for the installed MCP workflows, without network/model downloads."""
import asyncio
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

import numpy as np
import pytest
from sqlalchemy import text

from continual_brain.core.models import Lesson, Skill, Memory, LessonStatus, SkillStatus, RefinementAction
from continual_brain.core.refinement import RefinementEngine, RefinementProposal
from continual_brain.core.store import SQLiteStore
from continual_brain.core.research_scheduler import ResearchScheduler, ResearchTask, ResearchTrigger, ResearchJob, create_default_triggers
from continual_brain.query.continual_querier import ContinualQuerier, ContinualFAISSManager


class Encoder:
    def encode(self, texts, **kwargs):
        def vector(value):
            return [float('python' in value.lower()), float('docker' in value.lower()), 0.1]
        return np.asarray(vector(texts) if isinstance(texts, str) else [vector(t) for t in texts], dtype=np.float32)


@pytest.fixture
async def db(tmp_path):
    store = SQLiteStore(str(tmp_path / 'test.db'))
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def query(db, tmp_path, monkeypatch):
    monkeypatch.setattr('continual_brain.query.continual_querier.get_embedder', lambda: Encoder())
    return ContinualQuerier(db, ContinualFAISSManager(tmp_path / 'vectors.faiss'))


def proposal(kind='lesson', action=RefinementAction.CREATE, version=1, weight=.9, target='target', **values):
    return RefinementProposal(action, kind, target, version,
        {k: {'old': None, 'new': v} for k, v in values.items()}, 'regression', [], weight, .1)


@pytest.mark.parametrize('weight,auto,approved,expected', [(.8,True,False,False),(.85,True,False,True),(.99,False,False,False),(0,False,True,True)])
async def test_approval_policy(db, query, weight, auto, approved, expected):
    ok, _ = await RefinementEngine(db,query).apply_refinement(proposal(weight=weight,content='python'),auto_apply=auto,approved=approved)
    assert ok is expected
    assert bool(await db.get_lesson('target')) is expected


async def test_idempotent_concurrent_apply_and_rollback_skill(db, query):
    engine=RefinementEngine(db,query)
    p=proposal('skill',name='python',code='pass')
    results=await asyncio.gather(*(engine.apply_refinement(p,approved=True) for _ in range(2)))
    assert results[0] == results[1]
    assert (await db.get_skill('target')).status == SkillStatus.DRAFT
    assert await engine.rollback(results[0][1])
    assert await db.get_skill('target') is None
    assert await engine.rollback(results[0][1])
    with pytest.raises(ValueError,match='reverted'):
        await engine.apply_refinement(p,approved=True)


async def test_atomic_failure_leaves_no_partial_records(db,query,monkeypatch):
    engine=RefinementEngine(db,query)
    monkeypatch.setattr(db,'upsert_refinement',AsyncMock(side_effect=RuntimeError('disk failure')))
    with pytest.raises(RuntimeError):
        await engine.apply_refinement(proposal(content='python'),approved=True)
    assert await db.get_lesson('target') is None
    assert await db.list_proposals() == []
    assert await db.content_revision() == 0
    async with db.session() as session:
        assert (await session.execute(text('SELECT count(*) FROM snapshots'))).scalar() == 0


async def test_stale_apply_and_rollback_order(db,query):
    engine=RefinementEngine(db,query)
    _, first=await engine.apply_refinement(proposal(content='before'),approved=True)
    before=(await db.get_lesson('target')).to_dict()
    _, second=await engine.apply_refinement(proposal(action=RefinementAction.UPDATE,version=2,content='after'),approved=True)
    with pytest.raises(ValueError,match='Stale'):
        await engine.apply_refinement(proposal(action=RefinementAction.UPDATE,version=2,content='stale'),approved=True)
    with pytest.raises(ValueError,match='changed'):
        await engine.rollback(first)
    await engine.rollback(second)
    assert (await db.get_lesson('target')).to_dict() == before
    await engine.rollback(first)
    assert await db.get_lesson('target') is None


async def test_revision_index_restart_trust_filters_and_rollback(db,query):
    assert await query.query('python') == []
    memory=Memory(content='docker',tags=['persistent'],context={'trust_level':'untrusted'})
    await db.upsert_memory(memory)
    assert (await db.get_memory(memory.id)).tags == ['persistent']
    assert await query.query('docker') == []
    assert len(await query.query('docker',include_untrusted=True)) == 1
    engine=RefinementEngine(db,query)
    _, ref=await engine.apply_refinement(proposal(content='python'),approved=True)
    assert (await query.query('python'))[0]['entity'].id == 'target'
    reopened=ContinualQuerier(db,ContinualFAISSManager(query.faiss.index_path))
    assert (await reopened.query('python'))[0]['entity'].id == 'target'
    await engine.rollback(ref)
    assert await reopened.query('python') == []


async def test_index_write_failure_keeps_authoritative_database(db,query,monkeypatch):
    await db.upsert_lesson(Lesson(content='python',status=LessonStatus.ACCEPTED))
    monkeypatch.setattr(query.faiss,'save',lambda: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):
        await query.query('python')
    assert len(await db.list_lessons()) == 1
    assert query.faiss.generation is None


async def test_full_snapshot_not_truncated(db):
    async with db.transaction():
        for i in range(1001):
            await db.upsert_memory(Memory(id=f'm{i}',content='snapshot'))
    assert len((await db.export_state())['memories']) == 1001


def scheduler(db,tmp_path):
    web=AsyncMock()
    web.research_topic.return_value={'lessons_created':0}
    hybrid=AsyncMock()
    hybrid.query.return_value=[]
    return ResearchScheduler(db,hybrid,web,state_dir=str(tmp_path/'scheduler'))


async def test_scheduler_due_utc_cooldown_errors_and_defaults(db,tmp_path,monkeypatch):
    s=scheduler(db,tmp_path)
    now=datetime(2026,10,25,15,tzinfo=timezone.utc) # Sunday after 02:00
    monkeypatch.setattr('continual_brain.core.research_scheduler.utcnow',lambda:now)
    weekly=ResearchTask(name='weekly',topic='test',frequency='weekly')
    assert datetime.fromisoformat(s._calculate_next_run(weekly)) > now
    daily=ResearchTask(name='daily',topic='test')
    s.add_task(daily)
    daily.next_run=(now-timedelta(hours=1)).isoformat()
    await s._process_due_tasks()
    assert s.jobs[0].status == 'completed'
    s.add_trigger(ResearchTrigger(topic_pattern='python'))
    assert len(await s.check_triggers()) == 1
    assert await s.check_triggers() == []
    create_default_triggers(s)
    create_default_triggers(s)
    assert len(s.triggers) == 5
    s.web_researcher.research_topic.return_value={'error':'provider failed'}
    job=ResearchJob(topic='failed')
    s.jobs.append(job)
    await s._execute_job(job)
    assert job.status == 'failed'
    assert job.error == 'provider failed'


async def test_scheduler_concurrent_clients_and_single_owner(db,tmp_path):
    one=scheduler(db,tmp_path)
    two=scheduler(db,tmp_path)
    one.add_task(ResearchTask(name='one',topic='one',enabled=False))
    two.add_task(ResearchTask(name='two',topic='two',enabled=False))
    three=scheduler(db,tmp_path)
    assert len(three.tasks) == 2
    await one.start()
    await two.start()
    assert one.running and not two.running
    await one.stop()
    await two.start()
    assert two.running
    await two.stop()


async def test_scheduler_shutdown_tracks_running_job(db,tmp_path):
    s=scheduler(db,tmp_path)
    started=asyncio.Event()
    async def wait_forever(**kwargs):
        started.set()
        await asyncio.Event().wait()
    s.web_researcher.research_topic.side_effect=wait_forever
    job=ResearchJob(topic='slow')
    s.jobs.append(job)
    await s._process_pending_jobs()
    await started.wait()
    await s.stop()
    assert job.status == 'failed'
    assert not s._workers


async def test_mcp_dispatch_catalog_and_schema(db,tmp_path,query,monkeypatch):
    from continual_brain.mcp import server as m
    monkeypatch.setattr(m,'_store',db)
    monkeypatch.setattr(m,'_initialized',True)
    monkeypatch.setattr(m,'_refinement_engine',RefinementEngine(db,query))
    monkeypatch.setattr(m,'_scheduler',scheduler(db,tmp_path))
    monkeypatch.setenv('REFLEX_DATA_DIR',str(tmp_path))
    async def call(name,args, error=False):
        result=await m.call_tool(name,args)
        assert result.isError is error, result.content
        return json.loads(result.content[0].text)
    assert len(await m.list_tools()) >= 20
    await call('reflex_query',{'query':'test','top_k':-1},True)
    await call('reflex_query',{'query':'test','surprise':True},True)
    await call('reflex_get_lesson',{'lesson_id':'missing'},True)
    await call('reflex_get_skill',{'skill_id':'missing'},True)
    p=await call('reflex_propose_lesson',{'topic':'python','content':'Python data'})
    await call('reflex_apply_refinement',{'refinement_id':p['proposal_id'],'auto_apply':True},True)
    ref=await call('reflex_apply_refinement',{'refinement_id':p['proposal_id'],'approved':True})
    await call('reflex_get_refinement',{'refinement_id':ref['refinement_id']})
    await call('reflex_snapshot',{'label':'snapshot'})
    await call('reflex_list_lessons',{})
    await call('reflex_list_skills',{})
    await call('reflex_list_proposals',{})
    await call('reflex_status',{})
    await call('reflex_remember',{'content':'Docker preference','tags':['test']})
    task=await call('reflex_scheduler_add_task',{'name':'test','topic':'python'})
    await call('reflex_scheduler_list_tasks',{})
    await call('reflex_scheduler_remove_task',{'task_id':task['task_id']})
    await call('reflex_scheduler_add_trigger',{'name':'test','topic_pattern':'python','enabled':False})
    await call('reflex_scheduler_stats',{})
    await call('reflex_scheduler_check_coverage',{'topic':'python'})
    await call('reflex_rollback',{'refinement_id':ref['refinement_id']})
    await call('reflex_scheduler_add_task',{'name':'bad','topic':'test','frequency':'custom'},True)


async def test_web_content_stays_untrusted_and_stdout_clean(db,monkeypatch,capsys):
    from continual_brain.core.web_researcher import WebResearcher,SearchResult,ExtractedContent
    researcher=WebResearcher(db)
    urls=['https://example.com/one','https://example.com/two']
    researcher.searcher.search=AsyncMock(return_value=[SearchResult(u,'title','snippet') for u in urls])
    researcher._extract_concurrent=AsyncMock(return_value=[ExtractedContent(u,'Title','python '*100,str(i),{'domain':'example.com'},'2026',[],{}) for i,u in enumerate(urls)])
    try:
        result=await researcher.research_topic('python')
        assert result['lessons_created'] == 0
        assert result['lessons_proposed'] == 1
        assert result['memories_created'] == 2
        assert (await db.get_proposal(result['proposal_ids'][0]))['proposal']['evidence_weight'] == 0
        assert all(m.context['trust_level']=='untrusted' for m in await db.list_memories())
        assert capsys.readouterr().out == ''
    finally:
        await researcher.close()


@pytest.mark.parametrize('url',['http://127.0.0.1','http://[::1]','http://169.254.169.254','file:///etc/passwd','http://user:pass@example.com','http://example.com:8000'])
async def test_web_rejects_local_destinations(url):
    from continual_brain.core.web_researcher import validate_public_url
    with pytest.raises(ValueError):
        await validate_public_url(url)


async def test_web_pins_dns_and_revalidates_redirect(monkeypatch):
    import httpx
    from continual_brain.core import web_researcher as web
    validator=AsyncMock(side_effect=['93.184.216.34',ValueError('private redirect')])
    monkeypatch.setattr(web,'validate_public_url',validator)
    seen=[]
    def handle(request):
        seen.append(request)
        return httpx.Response(302,headers={'location':'http://127.0.0.1/private'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError,match='private'):
            await web.fetch_public_text(client,'https://example.com/page')
    assert len(seen)==1
    assert seen[0].url.host=='93.184.216.34'
    assert seen[0].headers['host']=='example.com'
    assert seen[0].extensions['sni_hostname']=='example.com'
    assert validator.await_args_list[1].args[0]=='http://127.0.0.1/private'


async def test_legacy_memory_tags_migration(tmp_path):
    import sqlite3
    path=tmp_path/'legacy.db'
    store=SQLiteStore(str(path))
    await store.initialize()
    await store.upsert_memory(Memory(id='legacy',content='preserve this'))
    await store.close()
    with sqlite3.connect(path) as connection:
        connection.execute('ALTER TABLE memories DROP COLUMN tags')
    upgraded=SQLiteStore(str(path))
    try:
        await upgraded.initialize()
        memory=await upgraded.get_memory('legacy')
        assert memory.content=='preserve this' and memory.tags==[]
        memory.tags=['migrated']
        await upgraded.upsert_memory(memory)
        assert (await upgraded.get_memory('legacy')).tags==['migrated']
        await upgraded.initialize()
    finally:
        await upgraded.close()

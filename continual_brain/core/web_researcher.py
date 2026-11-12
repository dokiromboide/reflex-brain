"""
Web Researcher - Automated web research for knowledge enrichment.
Searches web, extracts content, creates lessons/skills/memories.
"""
from __future__ import annotations
import asyncio
import json
import hashlib
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Dict, Optional
from urllib.parse import urlparse, urljoin, parse_qs

import httpx
from bs4 import BeautifulSoup

from continual_brain.core.models import Lesson, Skill, Memory, LessonStatus, SkillStatus, MemoryType
from continual_brain.core.store import SQLiteStore
from continual_brain.core.evidence import EntityExtractor, PatternMatcher


logger = logging.getLogger(__name__)

async def validate_public_url(url):
    import ipaddress
    import socket
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Only public HTTP(S) URLs without credentials are allowed")
    if parsed.port not in (None, 80, 443):
        raise ValueError("Nonstandard HTTP ports are not allowed")
    addresses = await asyncio.get_running_loop().getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Private, local and reserved network destinations are not allowed")
    return sorted({item[4][0] for item in addresses}, key=lambda address: ipaddress.ip_address(address).version)[0]

async def fetch_public_text(client, url, max_bytes=2_000_000):
    for _ in range(6):
        address = await validate_public_url(url)
        origin = httpx.URL(url)
        # Connect to the validated IP; preserve Host and TLS identity. This avoids
        # a second DNS lookup changing a public destination into a private one.
        async with client.stream("GET", origin.copy_with(host=address), follow_redirects=False,
                                 extensions={"sni_hostname": origin.host},
                                 headers={"Host": origin.netloc.decode("ascii"),
                                          "User-Agent": "ReflexBrain/0.2 (+https://github.com/dokiromboide/reflex-brain)"}) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers["location"])
                continue
            response.raise_for_status()
            if not any(kind in response.headers.get("content-type", "") for kind in ("text/html", "text/plain")):
                raise ValueError("Unsupported content type")
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError("Response exceeds download limit")
                chunks.append(chunk)
            return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace"), url
    raise ValueError("Too many redirects")

@dataclass
class SearchResult:
    """Result from web search."""
    url: str
    title: str
    snippet: str
    score: float = 0.0


@dataclass
class ExtractedContent:
    """Extracted and processed web content."""
    url: str
    title: str
    content: str
    content_hash: str
    metadata: Dict[str, Any]
    extracted_at: str
    entities: List[Dict]
    patterns: Dict[str, List]


@dataclass
class ResearchProposal:
    """Proposed knowledge from research."""
    type: str  # "lesson" | "skill" | "memory"
    title: str
    content: str
    evidence: List[Dict]
    confidence: float
    source_urls: List[str]
    cluster_id: Optional[str]
    tags: List[str]


class WebSearcher:
    """Searches web using multiple providers."""
    
    def __init__(self, provider: str = "duckduckgo"):
        self.provider = provider
        self.client = httpx.AsyncClient(timeout=20.0, follow_redirects=False, trust_env=False)
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search web and return results."""
        if self.provider == "duckduckgo":
            return await self._search_duckduckgo(query, max_results)
        elif self.provider == "bing":
            return await self._search_bing(query, max_results)
        else:
            return await self._search_duckduckgo(query, max_results)
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search via DuckDuckGo HTML (no API key needed)."""
        try:
            from urllib.parse import quote
            url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            html, _ = await fetch_public_text(self.client, url)
            soup = BeautifulSoup(html, "html.parser")
            results = []
            
            for result in soup.select(".result")[:max_results]:
                title_elem = result.select_one(".result__title")
                snippet_elem = result.select_one(".result__snippet")
                url_elem = result.select_one("a.result__a")
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    url = urljoin("https://html.duckduckgo.com", url_elem.get("href", "")) if url_elem else ""
                    url = parse_qs(urlparse(url).query).get("uddg", [url])[0]
                    if not url:
                        continue
                    
                    results.append(SearchResult(
                        url=url,
                        title=title,
                        snippet=snippet,
                        score=1.0 - (len(results) * 0.1)
                    ))
            
            return results
        except Exception as e:
            logger.warning("DuckDuckGo search failed: %s", e)
            return []
    
    async def _search_bing(self, query: str, max_results: int) -> List[SearchResult]:
        """Placeholder for Bing API (requires API key)."""
        # Would use Bing Web Search API here
        return []
    
    async def close(self):
        await self.client.aclose()


class ContentExtractor:
    """Extracts and processes content from web pages."""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.pattern_matcher = PatternMatcher()
        self.client = httpx.AsyncClient(timeout=20.0, follow_redirects=False, trust_env=False)
    
    async def extract(self, url: str) -> Optional[ExtractedContent]:
        """Extract content from URL."""
        try:
            html, url = await fetch_public_text(self.client, url)
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove script/style/nav/footer
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.get_text(strip=True)
            elif soup.h1:
                title = soup.h1.get_text(strip=True)
            
            # Extract main content
            content_selectors = [
                "main", "article", ".content", ".post-content", 
                ".entry-content", "#content", ".main-content"
            ]
            content = ""
            for selector in content_selectors:
                elem = soup.select_one(selector)
                if elem:
                    content = elem.get_text(separator="\n", strip=True)
                    break
            
            if not content:
                content = soup.get_text(separator="\n", strip=True)
            
            # Clean content
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            content = "\n".join(lines)
            
            if len(content) < 200:  # Too short
                return None
            
            # Limit content length
            content = content[:50000]
            
            # Generate hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            # Extract entities and patterns
            entities = self.entity_extractor.extract(content)
            patterns = self.pattern_matcher.match(content)
            
            # Metadata
            domain = urlparse(url).netloc
            metadata = {
                "domain": domain,
                "url": url,
                "content_length": len(content),
                "extracted_at": datetime.utcnow().isoformat() + "Z"
            }
            
            return ExtractedContent(
                url=url,
                title=title,
                content=content,
                content_hash=content_hash,
                metadata=metadata,
                extracted_at=datetime.utcnow().isoformat() + "Z",
                entities=[{"text": e.text, "label": e.label, "confidence": e.confidence} for e in entities],
                patterns=patterns
            )
            
        except Exception as e:
            logger.warning("Extraction failed for %s: %s", url, e)
            return None
    
    async def close(self):
        await self.client.aclose()


class KnowledgeSynthesizer:
    """Synthesizes extracted content into structured knowledge proposals."""
    
    def __init__(self, store: SQLiteStore):
        self.store = store
        self.entity_extractor = EntityExtractor()
        self.pattern_matcher = PatternMatcher()
    
    def synthesize_lessons(self, extracted: List[ExtractedContent], topic: str) -> List[ResearchProposal]:
        """Create lesson proposals from extracted content."""
        proposals = []
        
        # Group by domain/topic
        by_domain = {}
        for ext in extracted:
            domain = ext.metadata.get("domain", "unknown")
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(ext)
        
        for domain, contents in by_domain.items():
            if len(contents) < 2:
                continue  # Need multiple sources for synthesis
            
            # Combine content
            combined_content = "\n\n---\n".join([
                f"Source: {c.url}\n{c.content[:2000]}" for c in contents
            ])
            
            # Generate title
            title = f"{topic}: {domain}"
            if len(title) > 100:
                title = title[:97] + "..."
            
            # Evidence from all sources
            evidence = []
            for c in contents:
                evidence.append({
                    "source": f"web:{c.url}",
                    "quote": c.content[:500],
                    "weight": 0.7,
                    "title": c.title,
                    "domain": c.metadata.get("domain")
                })
            
            # Extract tags from entities
            all_entities = []
            for c in contents:
                all_entities.extend(c.entities)
            
            tags = list(set([
                e["label"].lower() for e in all_entities 
                if e["confidence"] > 0.7
            ]))
            
            # Add topic-based tags
            topic_tags = {
                "dian": ["dian", "factura", "iva", "facturación", "colombia"],
                "agentes": ["agente", "agent", "rlm", "subagent", "harness", "ai"],
                "novela": ["novela", "personaje", "capítulo", "escena", "trama", "escritura"],
                "config": ["config", "configuración", "hermes", "mcp", "setting"],
                "ingresos": ["ingreso", "income", "freelance", "workana", "cliente"],
                "graphrag": ["graphrag", "brain", "faiss", "embedding", "cluster", "louvain"],
                "marketing": ["marketing", "seo", "contenido", "redes", "social"],
                "python": ["python", "py", "pip", "venv", "asyncio", "fastapi"],
                "docker": ["docker", "container", "compose", "kubernetes", "k8s"],
            }
            
            topic_lower = topic.lower()
            for tag, keywords in topic_tags.items():
                if any(kw in topic_lower for kw in keywords):
                    tags.append(tag)
            
            # Determine cluster
            cluster_id = topic_lower.replace(" ", "_").replace("/", "_")
            
            proposals.append(ResearchProposal(
                type="lesson",
                title=title,
                content=combined_content[:10000],
                evidence=evidence,
                confidence=0.65,  # Slightly higher than single source
                source_urls=[c.url for c in contents],
                cluster_id=cluster_id,
                tags=tags
            ))
        
        return proposals
    
    def synthesize_skills(self, extracted: List[ExtractedContent]) -> List[ResearchProposal]:
        """Create skill proposals from code/tutorial content."""
        proposals = []
        
        for ext in extracted:
            content = ext.content.lower()
            
            # Detect code/tutorial patterns
            if any(kw in content for kw in ["def ", "function ", "class ", "import ", "from ", "async def"]):
                # Looks like code
                # Could extract and create a skill
                pass
            
            if any(kw in content for kw in ["tutorial", "how to", "step by step", "guide"]):
                # Tutorial content - could create a skill
                pass
        
        return proposals
    
    def synthesize_memories(self, extracted: List[ExtractedContent], topic: str) -> List[ResearchProposal]:
        """Create episodic memories from research."""
        proposals = []
        
        for ext in extracted:
            # Create memory for each significant source
            proposals.append(ResearchProposal(
                type="memory",
                title=f"Research: {ext.title[:80]}",
                content=f"Source: {ext.url}\n\n{ext.content[:3000]}",
                evidence=[{
                    "source": f"web:{ext.url}",
                    "quote": ext.content[:500],
                    "weight": 0.8,
                    "title": ext.title,
                    "domain": ext.metadata.get("domain")
                }],
                confidence=0.7,
                source_urls=[ext.url],
                cluster_id=topic.lower().replace(" ", "_"),
                tags=["research", "web-source"]
            ))
        
        return proposals


class WebResearcher:
    """Main orchestrator for automated web research."""
    
    def __init__(self, store: SQLiteStore, max_concurrent: int = 3):
        self.store = store
        self.searcher = WebSearcher()
        self.extractor = ContentExtractor()
        self.synthesizer = KnowledgeSynthesizer(store)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def research_topic(
        self, 
        topic: str, 
        max_sources: int = 10,
        create_lessons: bool = True,
        create_memories: bool = True
    ) -> Dict[str, Any]:
        """
        Full research pipeline: search → extract → synthesize → store.
        Returns summary of created knowledge.
        """
        logger.info("Researching: %s", topic)
        
        # 1. Search
        search_results = await self.searcher.search(topic, max_sources)
        logger.info("Found %s sources", len(search_results))
        
        if not search_results:
            return {"topic": topic, "lessons_created": 0, "memories_created": 0, "error": "No search results"}
        
        # 2. Extract content (concurrent)
        extracted = await self._extract_concurrent(search_results[:max_sources])
        valid_extracted = list({e.content_hash: e for e in extracted if isinstance(e, ExtractedContent)}.values())
        logger.info("Extracted %s valid pages", len(valid_extracted))
        
        if not valid_extracted:
            return {"topic": topic, "lessons_created": 0, "memories_created": 0, "error": "No valid content extracted"}
        
        # 3. Synthesize knowledge
        lessons_created = 0
        proposal_ids = []
        memories_created = 0
        
        if create_lessons:
            lesson_proposals = self.synthesizer.synthesize_lessons(valid_extracted, topic)
            for proposal in lesson_proposals:
                from continual_brain.core.refinement import RefinementProposal
                from continual_brain.core.models import RefinementAction
                lesson = self._create_lesson(proposal)
                fields = ("title", "content", "evidence", "confidence", "cluster_id", "tags")
                refinement = RefinementProposal(
                    action=RefinementAction.CREATE, target_type="lesson", target_id=lesson.id,
                    target_version=1, diff={key: {"old": None, "new": getattr(lesson, key)} for key in fields},
                    justification="Untrusted web research; review source URLs before approval",
                    evidence=proposal.evidence, evidence_weight=0.0, confidence_delta=lesson.confidence)
                await self.store.save_proposal(refinement.id, refinement.to_dict())
                proposal_ids.append(refinement.id)
        
        if create_memories:
            memory_proposals = self.synthesizer.synthesize_memories(valid_extracted, topic)
            for proposal in memory_proposals:
                memory = self._create_memory(proposal)
                await self.store.upsert_memory(memory)
                memories_created += 1
        
        # The query layer observes SQLite revisions and refreshes automatically.
        
        return {
            "topic": topic,
            "sources_found": len(search_results),
            "sources_extracted": len(valid_extracted),
            "lessons_created": lessons_created,
            "lessons_proposed": len(proposal_ids),
            "proposal_ids": proposal_ids,
            "trust_level": "untrusted",
            "memories_created": memories_created,
            "source_urls": [r.url for r in search_results[:max_sources]]
        }
    
    async def _extract_concurrent(self, search_results: List[SearchResult]) -> List[Optional[ExtractedContent]]:
        """Extract content from multiple URLs concurrently."""
        
        async def extract_with_semaphore(result: SearchResult) -> Optional[ExtractedContent]:
            async with self.semaphore:
                return await self.extractor.extract(result.url)
        
        tasks = [extract_with_semaphore(r) for r in search_results]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def _create_lesson(self, proposal: ResearchProposal) -> Lesson:
        """Create Lesson from research proposal."""
        import uuid
        return Lesson(
            id=f"lesson_{uuid.uuid4().hex[:12]}",
            version=1,
            title=proposal.title,
            content=proposal.content,
            evidence=proposal.evidence,
            confidence=proposal.confidence,
            status=LessonStatus.PROPOSED,
            cluster_id=proposal.cluster_id,
            tags=proposal.tags
        )
    
    def _create_memory(self, proposal: ResearchProposal) -> Memory:
        """Create Memory from research proposal."""
        import uuid
        return Memory(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            type=MemoryType.OBSERVATION,
            content=proposal.content,
            context={
                "source": "web_research",
                "trust_level": "untrusted",
                "topic": proposal.title,
                "source_urls": proposal.source_urls
            },
            importance=proposal.confidence,
            decay_rate=0.005,  # Slow decay for research
            cluster_id=proposal.cluster_id,
            tags=proposal.tags
        )
    
    async def _rebuild_index(self):
        """Rebuild FAISS index after new knowledge."""
        from continual_brain.query.continual_querier import ContinualFAISSManager, ContinualQuerier
        faiss_mgr = ContinualFAISSManager()
        querier = ContinualQuerier(self.store, faiss_mgr)
        await querier.rebuild_index()
    
    async def close(self):
        await self.searcher.close()
        await self.extractor.close()


async def research_topic(
    store: SQLiteStore,
    topic: str,
    max_sources: int = 10,
    create_lessons: bool = True,
    create_memories: bool = True
) -> Dict[str, Any]:
    """Convenience function for one-off research."""
    researcher = WebResearcher(store)
    try:
        return await researcher.research_topic(topic, max_sources, create_lessons, create_memories)
    finally:
        await researcher.close()

"""
Migration utilities - GraphRAG brain to Continual Brain.
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import asdict

from continual_brain.core.models import Lesson, LessonStatus
from continual_brain.core.store import SQLiteStore


class GraphRAGMigrator:
    """Migrates existing GraphRAG brain nodes to Continual Brain lessons."""
    
    def __init__(self, graph_dir: Path, store: SQLiteStore):
        self.graph_dir = Path(graph_dir)
        self.nodes_dir = self.graph_dir / "nodes"
        self.edges_dir = self.graph_dir / "edges"
        self.cluster_labels_path = self.graph_dir / "cluster_labels.json"
        self.store = store
        
        # Load cluster labels if available
        self.cluster_labels = {}
        if self.cluster_labels_path.exists():
            with open(self.cluster_labels_path, 'r', encoding='utf-8') as f:
                self.cluster_labels = json.load(f)
    
    def migrate(self, confidence_base: float = 0.6) -> Dict[str, int]:
        """
        Migrate all labeled nodes to lessons.
        Returns count of migrated items by type.
        """
        stats = {"lessons_created": 0, "skipped": 0, "errors": 0}
        
        if not self.nodes_dir.exists():
            raise ValueError(f"Nodes directory not found: {self.nodes_dir}")
        
        # Process each node file
        for node_file in self.nodes_dir.glob("*.json"):
            try:
                with open(node_file, 'r', encoding='utf-8') as f:
                    node = json.load(f)
                
                # Only migrate nodes with semantic labels (knowledge nodes)
                if "semantic_label" not in node:
                    stats["skipped"] += 1
                    continue
                
                # Skip conversation nodes unless they have substantial content
                if node.get("type") == "conversation_node":
                    content = node.get("properties", {}).get("content", "")
                    if len(content) < 500:  # Only substantial conversations
                        stats["skipped"] += 1
                        continue
                
                lesson = self._node_to_lesson(node, confidence_base)
                if lesson:
                    # Check if already exists (by title similarity)
                    existing = self._find_similar_lesson(lesson.title)
                    if existing:
                        # Merge evidence
                        self._merge_evidence(existing, lesson)
                        stats["skipped"] += 1
                    else:
                        self._save_lesson(lesson)
                        stats["lessons_created"] += 1
                        
            except Exception as e:
                print(f"Error migrating {node_file}: {e}")
                stats["errors"] += 1
        
        return stats
    
    def _node_to_lesson(self, node: Dict, confidence_base: float) -> Optional[Lesson]:
        """Convert a GraphRAG node to a Lesson."""
        node_id = node.get("id", "")
        node_type = node.get("type", "")
        label = node.get("label", "")
        content = node.get("content", node.get("text", ""))
        semantic_label = node.get("semantic_label", "")
        community = node.get("community", "")
        properties = node.get("properties", {})
        
        # Extract actual content from properties for conversation nodes
        if node_type == "conversation_node":
            content = properties.get("content", content)
        
        if not content or len(content) < 50:
            return None
        
        # Generate title
        title = self._generate_title(label, semantic_label, community, content)
        
        # Build evidence from node data
        evidence = [{
            "source": f"graphrag_node:{node_id}",
            "quote": content[:500],
            "weight": 0.7,
            "node_type": node_type,
            "community": community,
        }]
        
        # Add properties as evidence if relevant
        for key, value in properties.items():
            if key not in ("content", "role", "entities", "timestamp", "cluster_id") and value:
                evidence.append({
                    "source": f"graphrag_property:{key}",
                    "quote": str(value)[:300],
                    "weight": 0.5,
                })
        
        # Determine cluster_id from community or semantic_label
        cluster_id = community or semantic_label.lower().replace(" ", "_")
        
        lesson = Lesson(
            id=f"lesson_{uuid.uuid4().hex[:12]}",
            version=1,
            title=title,
            content=content[:10000],  # Limit content length
            evidence=evidence,
            confidence=confidence_base,
            status=LessonStatus.ACCEPTED,
            cluster_id=cluster_id,
            tags=self._extract_tags(content, semantic_label),
        )
        
        return lesson
    
    def _generate_title(self, label: str, semantic_label: str, community: str, content: str) -> str:
        """Generate a lesson title from node metadata."""
        # Use semantic label if available
        if semantic_label:
            base = semantic_label
        elif label and label not in ("Msg", "Node"):
            base = label
        else:
            # Extract from content
            base = content[:80].split('\n')[0].strip()
        
        # Clean up
        base = base.replace("Cluster_", "").replace("_", " ")
        if len(base) > 100:
            base = base[:97] + "..."
        
        return base
    
    def _extract_tags(self, content: str, semantic_label: str) -> List[str]:
        """Extract tags from content and label."""
        tags = []
        
        # From semantic label
        if semantic_label:
            tags.extend(semantic_label.lower().split("/"))
            tags.extend(semantic_label.lower().split(" "))
        
        # Keyword-based tags
        keyword_tags = {
            "dian": ["dian", "factura", "iva", "facturación"],
            "agentes": ["agente", "agent", "rlm", "subagent", "harness"],
            "novela": ["novela", "personaje", "capítulo", "escena", "trama"],
            "config": ["config", "configuración", "hermes", "mcp", "setting"],
            "ingresos": ["ingreso", "income", "freelance", "workana", "cliente"],
            "graphrag": ["graphrag", "brain", "faiss", "embedding", "cluster", "louvain"],
            "marketing": ["marketing", "seo", "contenido", "redes", "social"],
            "python": ["python", "py", "pip", "venv", "asyncio"],
            "docker": ["docker", "container", "compose", "kubernetes"],
        }
        
        content_lower = content.lower()
        for tag, keywords in keyword_tags.items():
            if any(kw in content_lower for kw in keywords):
                tags.append(tag)
        
        # Deduplicate
        return list(set(tags))
    
    def _find_similar_lesson(self, title: str) -> Optional[Lesson]:
        """Find existing lesson with similar title."""
        # Simple approach: check if any lesson has very similar title
        # In production, use semantic search
        import asyncio
        
        async def _search():
            lessons = await self.store.list_lessons(limit=200)
            for lesson in lessons:
                # Simple similarity: common words
                title_words = set(title.lower().split())
                lesson_words = set(lesson.title.lower().split())
                if title_words & lesson_words and len(title_words & lesson_words) >= 2:
                    return lesson
            return None
        
        try:
            return asyncio.run(_search())
        except RuntimeError:
            # Already in event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _search())
                return future.result()
    
    def _merge_evidence(self, existing: Lesson, new: Lesson):
        """Merge evidence from new lesson into existing."""
        # Add new evidence
        for ev in new.evidence:
            # Avoid duplicates
            if not any(e.get("source") == ev.get("source") for e in existing.evidence):
                existing.evidence.append(ev)
        
        # Update confidence (slight boost)
        existing.confidence = min(1.0, existing.confidence + 0.05)
        existing.updated_at = datetime.utcnow().isoformat() + "Z"
        
        self._save_lesson(existing)
    
    def _save_lesson(self, lesson: Lesson):
        """Save lesson to store."""
        import asyncio
        
        async def _save():
            await self.store.upsert_lesson(lesson)
        
        try:
            asyncio.run(_save())
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _save())
                future.result()


def migrate_graphrag(graph_dir: str, db_path: str, confidence_base: float = 0.6) -> Dict[str, int]:
    """Convenience function for migration."""
    store = SQLiteStore(db_path)
    migrator = GraphRAGMigrator(Path(graph_dir), store)
    return migrator.migrate(confidence_base)
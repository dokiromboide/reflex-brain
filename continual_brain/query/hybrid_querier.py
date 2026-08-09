"""
Hybrid Querier - Combines GraphRAG (conversations) + Continual (lessons/skills/memories).
Unified query interface with source_type filtering.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from continual_brain.core.store import SQLiteStore
from continual_brain.query.brain_querier import BrainQuerier, FAISSManager
from continual_brain.query.continual_querier import ContinualQuerier, ContinualFAISSManager


@dataclass
class HybridQueryResult:
    """Unified result from hybrid query."""
    source_type: str  # "conversation" | "knowledge" | "lesson" | "skill" | "memory"
    entity_type: str  # detailed type
    content: str
    title: str
    score: float
    confidence: float
    metadata: Dict[str, Any]


class HybridQuerier:
    """
    Unified query interface combining:
    - GraphRAG: conversation nodes + knowledge nodes (from existing Hermes Brain)
    - Continual: lessons, skills, memories (new structured knowledge)
    """
    
    def __init__(
        self,
        store: SQLiteStore,
        # GraphRAG paths
        brain_nodes_dir: str = "brain/nodes",
        brain_edges_dir: str = "brain/edges",
        brain_faiss_index: str = "brain/brain_index.faiss",
        brain_faiss_map: str = "brain/brain_nodes_map.pkl",
        # Continual paths
        continual_faiss_index: str = "continual_index.faiss",
        continual_faiss_map: str = "continual_nodes_map.pkl",
    ):
        self.store = store
        
        # GraphRAG querier
        self.brain_faiss = FAISSManager(brain_faiss_index, brain_faiss_map)
        self.brain_querier = BrainQuerier(
            self.brain_faiss,
            Path(brain_nodes_dir),
            Path(brain_edges_dir),
        )
        
        # Continual querier
        self.continual_faiss = ContinualFAISSManager(continual_faiss_index, continual_faiss_map)
        self.continual_querier = ContinualQuerier(store, self.continual_faiss)
    
    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        source_types: Optional[List[str]] = None,  # ["conversation", "knowledge", "lesson", "skill", "memory"]
        expand_depth: int = 1,
        min_confidence: float = 0.0,
    ) -> List[HybridQueryResult]:
        """
        Execute hybrid query across all memory sources.
        """
        all_results = []
        
        # Determine which sources to query
        query_conversations = source_types is None or "conversation" in source_types
        query_knowledge = source_types is None or "knowledge" in source_types
        query_lessons = source_types is None or "lesson" in source_types
        query_skills = source_types is None or "skill" in source_types
        query_memories = source_types is None or "memory" in source_types
        
        # 1. Query GraphRAG (conversations + knowledge)
        if query_conversations or query_knowledge:
            brain_results = await self._query_brain(query_text, top_k * 2, expand_depth)
            for r in brain_results:
                if r["entity_type"] == "conversation_node" and not query_conversations:
                    continue
                if r["entity_type"] != "conversation_node" and not query_knowledge:
                    continue
                all_results.append(HybridQueryResult(
                    source_type="conversation" if r["entity_type"] == "conversation_node" else "knowledge",
                    entity_type=r["entity_type"],
                    content=r["content"],
                    title=r["title"],
                    score=r["combined_score"],
                    confidence=r["confidence"],
                    metadata=r["metadata"],
                ))
        
        # 2. Query Continual (lessons, skills, memories)
        continual_source_types = []
        if query_lessons:
            continual_source_types.append("lesson")
        if query_skills:
            continual_source_types.append("skill")
        if query_memories:
            continual_source_types.append("memory")
        
        if continual_source_types:
            continual_results = await self.continual_querier.query(
                query_text,
                top_k=top_k * 2,
                source_types=continual_source_types,
                min_confidence=min_confidence,
            )
            for r in continual_results:
                all_results.append(HybridQueryResult(
                    source_type=r["type"],
                    entity_type=r["type"],
                    content=self._extract_content(r["entity"], r["type"]),
                    title=self._extract_title(r["entity"], r["type"]),
                    score=r["combined_score"],
                    confidence=r["confidence"],
                    metadata={
                        "status": getattr(r["entity"], "status", None).value if hasattr(r["entity"], "status") else None,
                        "cluster_id": getattr(r["entity"], "cluster_id", None),
                        "tags": getattr(r["entity"], "tags", []),
                    },
                ))
        
        # 3. Sort by score and return top_k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    async def _query_brain(self, query_text: str, top_k: int, expand_depth: int) -> List[Dict]:
        """Query GraphRAG brain and return structured results."""
        # Use brain_querier but we need structured output
        # For now, do vector search directly
        from continual_brain.query.brain_querier import get_embedder, combined_node_score, load_json, node_path, CONTINUAL_TYPES
        
        embedder = get_embedder()
        query_emb = embedder.encode(query_text)
        candidates = self.brain_faiss.search(query_emb, top_k * 3)
        
        results = []
        for node_id, vector_score in candidates:
            data = load_json(node_path(Path(self.brain_querier.nodes_dir), node_id))
            if not data:
                continue
            
            node_type = data.get("type", "unknown")
            combined = combined_node_score(vector_score, node_type, data)
            
            # Extract content
            if node_type == "conversation_node":
                content = data.get("properties", {}).get("content", "")
                title = f"[{data.get('properties', {}).get('role', '?')}] {data.get('label', node_id)}"
            else:
                content = data.get("content", data.get("text", data.get("label", "")))
                title = data.get("label", node_id)
            
            results.append({
                "entity_type": node_type,
                "content": content,
                "title": title,
                "combined_score": combined,
                "confidence": max(0, combined),  # proxy
                "metadata": {
                    "semantic_label": data.get("semantic_label"),
                    "community": data.get("community"),
                    "cluster_id": data.get("cluster_id"),
                },
            })
        
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_k]
    
    def _extract_content(self, entity, entity_type: str) -> str:
        if entity_type == "lesson":
            return entity.content
        elif entity_type == "skill":
            return f"{entity.description}\n\n{entity.code}"
        elif entity_type == "memory":
            return entity.content
        return ""
    
    def _extract_title(self, entity, entity_type: str) -> str:
        if entity_type == "lesson":
            return entity.title
        elif entity_type == "skill":
            return entity.name
        elif entity_type == "memory":
            return f"{entity.type.value}: {entity.content[:60]}..."
        return ""
    
    def format_results(self, results: List[HybridQueryResult]) -> str:
        """Format hybrid results as markdown context."""
        if not results:
            return "No relevant context found in any memory source."
        
        lines = ["## Hybrid Memory Context\n"]
        
        # Group by source type
        by_source = {}
        for r in results:
            if r.source_type not in by_source:
                by_source[r.source_type] = []
            by_source[r.source_type].append(r)
        
        source_labels = {
            "lesson": "📚 Lessons (Continual)",
            "skill": "⚙️ Skills (Continual)", 
            "memory": "🧠 Memories (Continual)",
            "knowledge": "📖 Knowledge Nodes (GraphRAG)",
            "conversation": "💬 Conversations (GraphRAG)",
        }
        
        for source_type, label in source_labels.items():
            if source_type not in by_source:
                continue
            
            lines.append(f"### {label}\n")
            for i, r in enumerate(by_source[source_type], 1):
                lines.append(f"**{i}. {r.title}**")
                lines.append(f"*Score: {r.score:.3f} | Confidence: {r.confidence:.2f}*")
                lines.append(f"{r.content[:1500]}")
                if r.metadata:
                    meta_str = ", ".join(f"{k}: {v}" for k, v in r.metadata.items() if v)
                    if meta_str:
                        lines.append(f"*{meta_str}*")
                lines.append("")
        
        return "\n".join(lines)
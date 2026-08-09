"""
Continual Brain Querier - Queries lessons, skills, and memories.
Prioritizes high-confidence, production-ready knowledge.
"""
from __future__ import annotations
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from continual_brain.core.store import SQLiteStore
from continual_brain.core.models import Lesson, Skill, Memory, LessonStatus, SkillStatus

# Lazy imports
_embedder = None
_faiss = None
_np = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("REFLEX_EMBED_MODEL", "all-MiniLM-L6-v2")
        _embedder = SentenceTransformer(model_name)
    return _embedder

def get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss

def get_np():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


EMBED_DIM = 384


class ContinualFAISSManager:
    """FAISS index for continual memory (lessons, skills, memories)."""
    
    def __init__(self, index_path: str = "continual_index.faiss", map_path: str = "continual_nodes_map.pkl"):
        self.index_path = Path(index_path)
        self.map_path = Path(map_path)
        self.index = None
        self.id_map = []  # list of (entity_type, entity_id)
        self._load()
    
    def _load(self):
        if self.index_path.exists():
            self.index = get_faiss().read_index(str(self.index_path))
            import pickle
            with open(self.map_path, "rb") as f:
                self.id_map = pickle.load(f)
        else:
            self.index = get_faiss().IndexFlatIP(EMBED_DIM)
            self.id_map = []
    
    def save(self):
        get_faiss().write_index(self.index, str(self.index_path))
        import pickle
        with open(self.map_path, "wb") as f:
            pickle.dump(self.id_map, f)
    
    def add(self, entity_type: str, entity_id: str, embedding: np.ndarray):
        faiss_id = self.index.ntotal
        self.index.add(get_np().array(embedding).reshape(1, -1).astype(get_np().float32))
        self.id_map.append((entity_type, entity_id))
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, str, float]]:
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(
            get_np().array(query_embedding).reshape(1, -1).astype(get_np().float32), 
            top_k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.id_map):
                entity_type, entity_id = self.id_map[idx]
                results.append((entity_type, entity_id, float(score)))
        return results
    
    def rebuild_from_store(self, store: SQLiteStore):
        """Rebuild index from database."""
        import asyncio
        
        async def _rebuild():
            self.index = get_faiss().IndexFlatIP(EMBED_DIM)
            self.id_map = []
            
            embedder = get_embedder()
            
            # Lessons
            lessons = await store.list_lessons(status=None, limit=10000)
            for lesson in lessons:
                text = f"{lesson.title}\n{lesson.content}"
                emb = embedder.encode(text)
                self.add("lesson", lesson.id, emb)
            
            # Skills
            skills = await store.list_skills(status=None, limit=10000)
            for skill in skills:
                text = f"{skill.name}\n{skill.description}\n{skill.code}"
                emb = embedder.encode(text)
                self.add("skill", skill.id, emb)
            
            # Memories
            memories = await store.list_memories(limit=10000)
            for memory in memories:
                text = f"{memory.content}\n{json.dumps(memory.context)}"
                emb = embedder.encode(text)
                self.add("memory", memory.id, emb)
            
            self.save()
        
        try:
            asyncio.run(_rebuild())
        except RuntimeError:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _rebuild())
                future.result()


import json


class ContinualQuerier:
    """
    Query layer for continual memory (lessons, skills, memories).
    Prioritizes high-confidence, production-ready knowledge.
    """
    
    def __init__(self, store: SQLiteStore, faiss_mgr: Optional[ContinualFAISSManager] = None):
        self.store = store
        self.faiss = faiss_mgr or ContinualFAISSManager()
    
    async def query(
        self, 
        query_text: str, 
        top_k: int = 5,
        source_types: Optional[List[str]] = None,  # ["lesson", "skill", "memory"]
        min_confidence: float = 0.0,
        status_filter: Optional[str] = None,  # "accepted", "production", etc.
    ) -> List[Dict[str, Any]]:
        """
        Query continual memory.
        Returns list of dicts with entity data + score.
        """
        # 1. Vector search
        embedder = get_embedder()
        query_emb = embedder.encode(query_text)
        candidates = self.faiss.search(query_emb, top_k * 3)
        
        # 2. Fetch and filter
        results = []
        for entity_type, entity_id, vector_score in candidates:
            if source_types and entity_type not in source_types:
                continue
            
            entity = await self._fetch_entity(entity_type, entity_id)
            if not entity:
                continue
            
            # Confidence filter
            if entity.confidence < min_confidence:
                continue
            
            # Status filter
            if status_filter:
                entity_status = getattr(entity, 'status', None)
                if entity_status and entity_status.value != status_filter:
                    continue
            
            # Combined score
            combined = vector_score * (0.5 + entity.confidence)
            
            results.append({
                "type": entity_type,
                "entity": entity,
                "vector_score": vector_score,
                "confidence": entity.confidence,
                "combined_score": combined,
            })
        
        # 3. Sort by combined score
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_k]
    
    async def _fetch_entity(self, entity_type: str, entity_id: str):
        if entity_type == "lesson":
            return await self.store.get_lesson(entity_id)
        elif entity_type == "skill":
            return await self.store.get_skill(entity_id)
        elif entity_type == "memory":
            return await self.store.get_memory(entity_id)
        return None
    
    async def get_lesson(self, lesson_id: str) -> Optional[Dict]:
        lesson = await self.store.get_lesson(lesson_id)
        if lesson:
            return {"type": "lesson", "entity": lesson}
        return None
    
    async def get_skill(self, skill_id: str) -> Optional[Dict]:
        skill = await self.store.get_skill(skill_id)
        if skill:
            return {"type": "skill", "entity": skill}
        return None
    
    async def get_memory(self, memory_id: str) -> Optional[Dict]:
        memory = await self.store.get_memory(memory_id)
        if memory:
            return {"type": "memory", "entity": memory}
        return None
    
    def format_results(self, results: List[Dict]) -> str:
        """Format results as markdown context."""
        if not results:
            return "No relevant knowledge found in continual memory."
        
        lines = ["## Relevant Knowledge from Continual Brain\n"]
        
        for i, result in enumerate(results, 1):
            entity = result["entity"]
            entity_type = result["type"]
            
            if entity_type == "lesson":
                lines.append(f"### {i}. 📚 Lesson: {entity.title}")
                lines.append(f"**Confidence**: {entity.confidence:.2f} | **Status**: {entity.status.value} | **Cluster**: {entity.cluster_id or 'N/A'}")
                lines.append(f"**Tags**: {', '.join(entity.tags) if entity.tags else 'N/A'}")
                lines.append(f"**Content**:\n{entity.content[:2000]}")
                if entity.evidence:
                    lines.append(f"**Evidence**: {len(entity.evidence)} sources")
            
            elif entity_type == "skill":
                lines.append(f"### {i}. ⚙️ Skill: {entity.name}")
                lines.append(f"**Confidence**: {entity.confidence:.2f} | **Status**: {entity.status.value}")
                lines.append(f"**Description**: {entity.description}")
                lines.append(f"**Interface**: {json.dumps(entity.interface, ensure_ascii=False)}")
                if entity.test_cases:
                    lines.append(f"**Tests**: {len(entity.test_cases)} cases")
            
            elif entity_type == "memory":
                lines.append(f"### {i}. 🧠 Memory: {entity.type.value}")
                lines.append(f"**Importance**: {entity.importance:.2f} | **Access Count**: {entity.access_count}")
                lines.append(f"**Content**:\n{entity.content[:1500]}")
                if entity.context:
                    lines.append(f"**Context**: {json.dumps(entity.context, ensure_ascii=False)[:500]}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    async def rebuild_index(self):
        """Rebuild FAISS index from database."""
        self.faiss.rebuild_from_store(self.store)
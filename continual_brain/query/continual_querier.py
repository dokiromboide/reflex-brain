"""SQLite-authoritative semantic retrieval with revision-aware derived FAISS indexes."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from continual_brain.core.store import SQLiteStore

logger = logging.getLogger(__name__)
_embedder = None
_faiss = None
EMBED_DIM = 384

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(os.getenv("REFLEX_EMBED_MODEL", "all-MiniLM-L6-v2"))
    return _embedder

def get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss

def get_np():
    return np


class ContinualFAISSManager:
    """One atomically replaced, non-pickle archive containing vectors, IDs and revision."""

    def __init__(self, index_path=None, map_path=None):
        root = Path(os.getenv("REFLEX_FAISS_PATH", ".")).expanduser()
        self.index_path = Path(index_path) if index_path else root / "continual_index.faiss"
        self.map_path = Path(map_path) if map_path else root / "continual_nodes_map.json"
        self.index = get_faiss().IndexFlatIP(EMBED_DIM)
        self.id_map = []
        self.generation = None
        self._load()

    def _load(self):
        if not self.index_path.exists():
            return
        try:
            with np.load(self.index_path, allow_pickle=False) as archive:
                index = get_faiss().deserialize_index(archive["index"])
                ids = json.loads(str(archive["ids"].item()))
                generation = json.loads(str(archive["generation"].item()))
            if len(ids) != index.ntotal or any(len(x) != 2 for x in ids):
                raise ValueError("Vector/ID count mismatch")
            self.index, self.id_map, self.generation = index, ids, generation
        except Exception:
            # Indexes are disposable. Never deserialize legacy pickle maps.
            logger.warning("Ignoring legacy or invalid continual index; it will rebuild from SQLite")

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=self.index_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, index=get_faiss().serialize_index(self.index),
                                    ids=np.asarray(json.dumps(self.id_map)),
                                    generation=np.asarray(json.dumps(self.generation)))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.index_path)
        finally:
            Path(name).unlink(missing_ok=True)

    def add(self, entity_type, entity_id, embedding):
        key = [entity_type, entity_id]
        if key in [list(x) for x in self.id_map]:
            raise ValueError("Duplicate vector ID")
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        get_faiss().normalize_L2(vector)
        self.index.add(vector)
        self.id_map.append(key)

    def search(self, query_embedding, top_k=5):
        if not self.index.ntotal:
            return []
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        get_faiss().normalize_L2(vector)
        scores, indices = self.index.search(vector, min(top_k, self.index.ntotal))
        return [(self.id_map[i][0], self.id_map[i][1], float(score))
                for score, i in zip(scores[0], indices[0]) if 0 <= i < len(self.id_map)]


class ContinualQuerier:
    def __init__(self, store: SQLiteStore, faiss_mgr=None):
        self.store = store
        self.faiss = faiss_mgr or ContinualFAISSManager()
        self._index_lock = asyncio.Lock()

    async def _ensure_index(self, force=False):
        async with self._index_lock:
            generation = [self.store.db_path, await self.store.content_revision(),
                          os.getenv("REFLEX_EMBED_MODEL", "all-MiniLM-L6-v2"), 1]
            if not force and self.faiss.generation == generation:
                return
            async with self.store.transaction():
                state = await self.store.export_state()
                generation[1] = await self.store.content_revision()
            texts, ids = [], []
            for kind, entries in state.items():
                for entry in entries:
                    if kind == "lessons":
                        content = entry["title"] + "\n" + entry["content"]
                    elif kind == "skills":
                        content = entry["name"] + "\n" + entry["description"] + "\n" + entry["code"]
                    else:
                        content = entry["content"]
                    texts.append(content)
                    ids.append([kind[:-1] if kind != "memories" else "memory", entry["id"]])
            # Initialize native ML modules on the main thread. Importing SciPy's
            # extension modules for the first time in a Windows worker can hang.
            # Encoding remains off the event loop after this one-time load.
            embedder = get_embedder() if texts else None
            def build():
                if texts:
                    vectors = np.asarray(embedder.encode(texts, batch_size=32,
                        show_progress_bar=False, normalize_embeddings=True), dtype=np.float32)
                    get_faiss().normalize_L2(vectors)
                    index = get_faiss().IndexFlatIP(vectors.shape[1])
                    index.add(vectors)
                else:
                    index = get_faiss().IndexFlatIP(EMBED_DIM)
                return index
            index = await asyncio.to_thread(build)
            previous = self.faiss.index, self.faiss.id_map, self.faiss.generation
            self.faiss.index, self.faiss.id_map, self.faiss.generation = index, ids, generation
            try:
                # Keep this short commit synchronous: no other coroutine sees a half-published index.
                self.faiss.save()
            except BaseException:
                self.faiss.index, self.faiss.id_map, self.faiss.generation = previous
                raise

    async def rebuild_index(self):
        await self._ensure_index(force=True)

    async def query(self, query_text, top_k=5, source_types=None, min_confidence=0.0,
                    status_filter=None, include_untrusted=False):
        await self._ensure_index()
        if not self.faiss.index.ntotal:
            return []
        embedder = get_embedder()
        vector = await asyncio.to_thread(lambda: embedder.encode(query_text, normalize_embeddings=True))
        # Filter before the final top-k, so unrelated/proposed entities cannot crowd accepted results out.
        candidates = self.faiss.search(vector, self.faiss.index.ntotal)
        results = []
        for kind, entity_id, score in candidates:
            if source_types and kind not in source_types:
                continue
            entity = await self._fetch_entity(kind, entity_id)
            if entity is None:
                continue
            status = getattr(entity, "status", None)
            if status_filter and (status is None or status.value != status_filter):
                continue
            if not status_filter and not include_untrusted:
                if kind == "lesson" and status.value != "accepted":
                    continue
                if kind == "skill" and status.value not in ("tested", "production"):
                    continue
                if kind == "memory" and entity.context.get("trust_level") == "untrusted":
                    continue
            confidence = entity.importance if kind == "memory" else entity.confidence
            if confidence < min_confidence:
                continue
            results.append({"type": kind, "entity": entity, "vector_score": score,
                            "confidence": confidence, "combined_score": score * (0.5 + confidence)})
        results.sort(key=lambda item: item["combined_score"], reverse=True)
        return results[:top_k]

    async def _fetch_entity(self, kind, entity_id):
        if kind not in ("lesson", "skill", "memory"):
            return None
        return await getattr(self.store, f"get_{kind}")(entity_id)

    async def get_lesson(self, entity_id):
        entity = await self.store.get_lesson(entity_id)
        return {"type": "lesson", "entity": entity} if entity else None

    async def get_skill(self, entity_id):
        entity = await self.store.get_skill(entity_id)
        return {"type": "skill", "entity": entity} if entity else None

    async def get_memory(self, entity_id):
        entity = await self.store.get_memory(entity_id)
        return {"type": "memory", "entity": entity} if entity else None

    def format_results(self, results: list[dict]) -> str:
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

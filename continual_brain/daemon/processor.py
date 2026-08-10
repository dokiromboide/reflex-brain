"""
Continuous Processor - Background daemon for passive memory capture.
Adapted from hermes-mcp-extensions skill Patterns 7, 8, 9, 10.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Lazy imports
_sqlite3 = None
_spacy = None
_sentence_transformers = None
_faiss = None
_nx = None
_watchdog = None

def get_sqlite3():
    global _sqlite3
    if _sqlite3 is None:
        import sqlite3
        _sqlite3 = sqlite3
    return _sqlite3

def get_spacy():
    global _spacy
    if _spacy is None:
        import spacy
        _spacy = spacy
    return _spacy

def get_sentence_transformers():
    global _sentence_transformers
    if _sentence_transformers is None:
        from sentence_transformers import SentenceTransformer
        _sentence_transformers = SentenceTransformer
    return _sentence_transformers

def get_faiss():
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss

def get_nx():
    global _nx
    if _nx is None:
        import networkx as nx
        _nx = nx
    return _nx

def get_watchdog():
    global _watchdog
    if _watchdog is None:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
        _watchdog = (Observer, FileSystemEventHandler)
    return _watchdog


logger = logging.getLogger(__name__)

# Constants
EMBED_DIM = 384
BATCH_SIZE = 32
SAVE_INTERVAL = 100  # FAISS save every N additions

# Directories
BRAIN_DIR = Path(os.getenv("REFLEX_FAISS_PATH", ".")) / "brain"
NODES_DIR = BRAIN_DIR / "nodes"
EDGES_DIR = BRAIN_DIR / "edges"
BRAIN_INDEX = BRAIN_DIR / "brain_index.faiss"
BRAIN_NODES_MAP = BRAIN_DIR / "brain_nodes_map.pkl"
CONTINUAL_INDEX = BRAIN_DIR / "continual_index.faiss"
CONTINUAL_MAP = BRAIN_DIR / "continual_nodes_map.pkl"
DAEMON_STATE_FILE = BRAIN_DIR / "daemon_state.json"

# Ensure directories exist
NODES_DIR.mkdir(parents=True, exist_ok=True)
EDGES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DaemonState:
    """Persistent daemon state."""
    last_message_id: int = 0
    processed_count: int = 0
    last_run: str = ""

    @classmethod
    def load(cls) -> DaemonState:
        if DAEMON_STATE_FILE.exists():
            import json
            with open(DAEMON_STATE_FILE) as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def save(self):
        import json
        with open(DAEMON_STATE_FILE, 'w') as f:
            json.dump({
                "last_message_id": self.last_message_id,
                "processed_count": self.processed_count,
                "last_run": self.last_run,
            }, f)


class SQLiteWatcher:
    """Watches Hermes state.db for new messages."""

    def __init__(self, db_path: Path, state: DaemonState):
        self.db_path = db_path
        self.state = state
        self._conn = None

    def connect(self):
        if self._conn is None:
            self._conn = get_sqlite3().connect(
                f"file:{self.db_path}?mode=ro",
                uri=True
            )
            self._conn.row_factory = get_sqlite3().Row

    def fetch_new_messages(self) -> list[dict]:
        self.connect()
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE id > ? ORDER BY id",
            (self.state.last_message_id,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


class BatchEmbedder:
    """Batches embedding calls for throughput."""

    def __init__(self, batch_size: int = BATCH_SIZE):
        self.batch_size = batch_size
        self._pending: list[tuple] = []  # (text, idx)
        self._completed: list[tuple] = []  # (idx, embedding)
        self._counter = 0

    def add(self, text: str) -> int:
        idx = self._counter
        self._counter += 1
        self._pending.append((text, idx))
        if len(self._pending) >= self.batch_size:
            self.flush()
        return idx

    def flush(self):
        if not self._pending:
            return

        texts = [t for t, _ in self._pending]
        embedder = get_sentence_transformers()(os.getenv("REFLEX_EMBED_MODEL", "all-MiniLM-L6-v2"))
        embeddings = embedder.encode(texts, batch_size=len(texts), show_progress_bar=False)

        for (_, idx), emb in zip(self._pending, embeddings):
            self._completed.append((idx, emb))

        self._pending.clear()

    def get(self, idx: int):
        for i, emb in self._completed:
            if i == idx:
                return emb
        raise IndexError(f"Embedding {idx} not ready")


class FAISSManager:
    """Manages FAISS index with batch saves."""

    def __init__(self, index_path: Path, map_path: Path):
        self.index_path = index_path
        self.map_path = map_path
        self.index = None
        self.id_map = []
        self._pending_adds = 0
        self._load()

    def _load(self):
        if self.index_path.exists():
            self.index = get_faiss().read_index(str(self.index_path))
            import pickle
            with open(self.map_path, "rb") as f:
                self.id_map = pickle.load(f)
            logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
        else:
            self.index = get_faiss().IndexFlatIP(EMBED_DIM)
            self.id_map = []
            logger.info("Created new FAISS index")

    def save(self):
        get_faiss().write_index(self.index, str(self.index_path))
        import pickle
        with open(self.map_path, "wb") as f:
            pickle.dump(self.id_map, f)

    def add(self, node_id: str, embedding):
        faiss_id = self.index.ntotal
        self.index.add(get_faiss().array(embedding).reshape(1, -1).astype(get_faiss().float32))
        if isinstance(self.id_map, list):
            self.id_map.append(node_id)
        else:
            self.id_map[faiss_id] = node_id
        self._pending_adds += 1
        if self._pending_adds >= SAVE_INTERVAL:
            self.save()
            self._pending_adds = 0

    def flush(self):
        if self._pending_adds > 0:
            self.save()
            self._pending_adds = 0

    def search(self, query_embedding, top_k: int = 5):
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(
            get_faiss().array(query_embedding).reshape(1, -1).astype(get_faiss().float32),
            top_k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if isinstance(self.id_map, list):
                if 0 <= idx < len(self.id_map):
                    results.append((self.id_map[idx], float(score)))
            elif idx in self.id_map:
                results.append((self.id_map[idx], float(score)))
        return results


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))[:60] or "node"


import re


def node_path(node_id: str) -> Path:
    return NODES_DIR / f"{_safe_name(node_id)}.json"


def edge_path(source: str, relation: str, target: str) -> Path:
    return EDGES_DIR / f"{_safe_name(source)}_{_safe_name(relation)}_{_safe_name(target)}.json"


def save_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


import json


def load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


class EntityExtractor:
    """Extracts entities from text using spaCy NER + regex."""

    CUSTOM_PATTERNS = [
        (r'\b(?:DIAN|IVA|RUT|NIT|factura\s+electr[oó]nica)\b', 'TAX_CONCEPT'),
        (r'\b(?:Hermes|MCP|Brain|GraphRAG|FAISS|Louvain)\b', 'TECH_STACK'),
        (r'\b(?:Tecnosfera|Avena\s+Cubana|Pop)\b', 'BRAND'),
        (r'\b(?:Workana|Upwork|Freelancer|Fiverr)\b', 'PLATFORM'),
        (r'\b(?:Kimi\s*K?3|GLM-?5\.?2|Nemotron|DeepSeek|Qwen)\b', 'MODEL'),
        (r'\b(?:vLLM|SGLang|TokenSpeed|OpenInterpreter|Prime\s+Agent)\b', 'TOOL'),
        (r'\b(?:RLM|Continual\s+Harness|subagent|skill)\b', 'CONCEPT'),
        (r'\b\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s*(?:USD|COP|EUR)\b', 'MONEY'),
        (r'\b(?:https?://|www\.)\S+\b', 'URL'),
        (r'\b[A-Za-z]:[\\/][\w\\/.-]+\b', 'PATH'),
        (r'\b\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+)?\b', 'VERSION'),
        (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'IP'),
    ]

    def __init__(self):
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            spacy = get_spacy()
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                self._nlp = spacy.blank("en")
        return self._nlp

    def extract(self, text: str) -> list[dict]:
        entities = []

        # spaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })

        # Custom regex patterns
        for pattern, label in self.CUSTOM_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                })

        # Deduplicate
        return self._deduplicate(entities)

    def _deduplicate(self, entities: list[dict]) -> list[dict]:
        if not entities:
            return []
        entities.sort(key=lambda e: (e["start"], -len(e["text"])))
        result = [entities[0]]
        for ent in entities[1:]:
            last = result[-1]
            if ent["start"] < last["end"]:
                if len(ent["text"]) > len(last["text"]):
                    result[-1] = ent
            else:
                result.append(ent)
        return result


class NodeWriter:
    """Writes nodes to filesystem graph."""

    def write_conversation_node(self, msg: dict, embedding_id: int) -> str:
        node_id = f"msg_{msg['id']}"
        node = {
            "id": node_id,
            "type": "conversation_node",
            "label": f"Msg {msg['id']}",
            "text": f"Msg {msg['id']}",
            "properties": {
                "content": msg.get("content", ""),
                "role": msg.get("role", "unknown"),
                "entities": [],  # Filled by processor
                "timestamp": msg.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            }
        }
        save_json(node_path(node_id), node)
        return node_id

    def write_entity_node(self, entity: dict, embedding_id: int) -> str:
        label = entity["label"]
        text = entity["text"]
        node_id = f"ent_{_safe_name(f'{label}_{text}')}"
        node = {
            "id": node_id,
            "type": "entity",
            "label": entity["label"],
            "text": entity["text"],
            "properties": {
                "entity_label": entity["label"],
            }
        }
        save_json(node_path(node_id), node)
        return node_id


class EdgeWriter:
    """Writes edges to filesystem graph."""

    def write_edge(self, source: str, relation: str, target: str, weight: float = 1.0):
        path = edge_path(source, relation, target)
        edge = {
            "source": source,
            "relation": relation,
            "target": target,
            "weight": weight,
        }
        save_json(path, edge)


class MessageProcessor:
    """Processes messages in batches: extract entities, create nodes/edges, embed."""

    def __init__(
        self,
        extractor: EntityExtractor,
        node_writer: NodeWriter,
        edge_writer: EdgeWriter,
        faiss_mgr: FAISSManager,
        batch_embedder: BatchEmbedder,
    ):
        self.extractor = extractor
        self.node_writer = node_writer
        self.edge_writer = edge_writer
        self.faiss = faiss_mgr
        self.batch_embedder = batch_embedder

    def process_batch(self, messages: list[dict]) -> list[dict]:
        results = []

        for msg in messages:
            try:
                # 1. Create conversation node
                conv_node_id = self.node_writer.write_conversation_node(msg, self.faiss.index.ntotal)

                # 2. Extract entities
                content = msg.get("content", "")
                entities = self.extractor.extract(content)

                # 3. Update node with entities
                node = load_json(node_path(conv_node_id))
                if node:
                    node["properties"]["entities"] = entities
                    save_json(node_path(conv_node_id), node)

                # 4. Create entity nodes and edges
                for entity in entities:
                    ent_node_id = self.node_writer.write_entity_node(entity, self.faiss.index.ntotal)
                    self.edge_writer.write_edge(conv_node_id, "MENTIONS", ent_node_id)

                    # Add to FAISS
                    self._add_to_faiss(ent_node_id, entity["text"])

                # 5. Add conversation node to FAISS
                self._add_to_faiss(conv_node_id, content)

                results.append({"msg_id": msg["id"], "node_id": conv_node_id, "entities": len(entities)})

            except Exception as e:
                logger.error(f"Error processing message {msg.get('id')}: {e}")
                results.append({"msg_id": msg.get("id"), "error": str(e)})

        # Flush embeddings
        self.batch_embedder.flush()
        self.faiss.flush()

        return results

    def _add_to_faiss(self, node_id: str, text: str):
        idx = self.batch_embedder.add(text)
        # We'll get the embedding after flush
        # For now, store mapping - actual embedding added in flush


class CommunityClassifier:
    """Louvain community detection on entity co-occurrence graph."""

    def run_clustering(self):
        G = get_nx().Graph()

        # Build graph from edges
        for edge_file in EDGES_DIR.glob("*.json"):
            edge = load_json(edge_file)
            if edge:
                G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))

        if len(G) < 3:
            return

        try:
            communities = list(get_nx().community.louvain_communities(G, weight="weight"))

            for i, comm in enumerate(communities):
                cluster_id = f"Cluster_{i}"
                for node_id in comm:
                    path = node_path(node_id)
                    data = load_json(path)
                    if data:
                        data["community"] = cluster_id
                        save_json(path, data)

            logger.info(f"Detected {len(communities)} communities")
        except Exception as e:
            logger.error(f"Clustering failed: {e}")


async def run_daemon(poll_interval: float = 3.0, batch_size: int = 50, enable_scheduler: bool = True):
    """Main daemon loop."""
    logger.info("Starting Reflex Brain daemon...")

    # Initialize
    state = DaemonState.load()
    db_path = Path(os.getenv("HERMES_STATE_DB", "C:/Users/USER/AppData/Local/hermes/state.db"))

    watcher = SQLiteWatcher(db_path, state)
    extractor = EntityExtractor()
    node_writer = NodeWriter()
    edge_writer = EdgeWriter()
    faiss_mgr = FAISSManager(BRAIN_INDEX, BRAIN_NODES_MAP)
    batch_embedder = BatchEmbedder(batch_size)
    processor = MessageProcessor(extractor, node_writer, edge_writer, faiss_mgr, batch_embedder)
    classifier = CommunityClassifier()

    # Initialize scheduler if enabled
    scheduler = None
    if enable_scheduler:
        try:
            from continual_brain.core.store import SQLiteStore
            from continual_brain.query.hybrid_querier import HybridQuerier
            from continual_brain.core.web_researcher import WebResearcher
            from continual_brain.core.research_scheduler import create_scheduler, create_default_triggers
            
            continual_db = Path(os.getenv("REFLEX_DB_PATH", "continual.db"))
            store = SQLiteStore(str(continual_db))
            await store.initialize()
            
            querier = HybridQuerier(
                store=store,
                brain_nodes_dir="brain/nodes",
                brain_edges_dir="brain/edges",
                brain_faiss_index="brain/brain_index.faiss",
                brain_faiss_map="brain/brain_nodes_map.pkl",
                continual_faiss_index="continual_index.faiss",
                continual_faiss_map="continual_nodes_map.pkl",
            )
            
            web_researcher = WebResearcher(store)
            scheduler = create_scheduler(store, querier, web_researcher)
            create_default_triggers(scheduler)
            await scheduler.start()
            logger.info("Research scheduler started with default triggers")
        except Exception as e:
            logger.warning(f"Could not start scheduler: {e}")

    logger.info(f"Daemon state: processed={state.processed_count}, last_msg_id={state.last_message_id}")

    try:
        while True:
            try:
                # Fetch new messages
                rows = watcher.fetch_new_messages()

                if rows:
                    # Process in batches
                    for i in range(0, len(rows), batch_size):
                        batch = rows[i:i+batch_size]
                        processor.process_batch(batch)

                        for row in batch:
                            state.last_message_id = row["id"]
                            state.processed_count += 1

                        state.last_run = datetime.utcnow().isoformat() + "Z"
                        state.save()

                    logger.info(f"Processed {len(rows)} messages. Total: {state.processed_count}")

                    # Run clustering every 50 messages
                    if state.processed_count % 50 == 0:
                        classifier.run_clustering()
                else:
                    logger.debug("No new messages")

                await asyncio.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Daemon error: {e}")
                await asyncio.sleep(poll_interval * 2)
    finally:
        if scheduler:
            await scheduler.stop()
            logger.info("Research scheduler stopped")

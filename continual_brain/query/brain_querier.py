"""
GraphRAG BrainQuerier - Adapted from hermes-mcp-extensions skill.
Quality-filtered semantic search over conversation nodes + knowledge nodes.
"""
from __future__ import annotations
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

# Lazy imports for heavy dependencies
_nlp = None
_embedder = None
_faiss = None
_np = None

def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

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

# Type priority (from skill Pattern 11)
HIGH_VALUE_TYPES = {
    'concept', 'process', 'principle', 'error_lesson', 'research_note',
    'organization', 'person', 'project', 'skill', 'decision', 'pattern',
    'lesson', 'skill', 'memory'  # Continual types
}
LOW_VALUE_TYPES = {'message', 'tool_call', 'session'}

# Tool output detection patterns (from skill Pattern 11)
TOOL_OUTPUT_INDICATORS = [
    '{"output":', '"exit_code"', '"stdout"', '"stderr"', '"duration_seconds"',
    '↪', '.jar', '.png', 'rw-r--r--', 'citadel', 'yung', 'kubejs', 'forge',
    'File Added', 'File Modified', 'File Deleted', 'ApprovalRequest',
    'pip install', 'npm install', 'cargo build', 'go build',
    'Traceback', 'Error:', 'Exception:', 'SyntaxError',
    'Loading weights', 'Batches:', 'it/s', 'HTTP Request:',
]


def content_quality_score(data: Dict[str, Any]) -> float:
    """
    Score content quality 0.0-2.0.
    Negative = tool output (penalized).
    """
    # Get content from properties.content for conversation_node, or content field
    content = ""
    if data.get("type") == "conversation_node":
        content = data.get("properties", {}).get("content", "")
    else:
        content = data.get("content", data.get("text", ""))
    
    if not content or not isinstance(content, str):
        return -1.0
    
    # Check for tool output indicators
    content_lower = content.lower()
    for indicator in TOOL_OUTPUT_INDICATORS:
        if indicator.lower() in content_lower:
            return -1.0  # Tool output penalty
    
    # Score by length
    length = len(content)
    if length < 50:
        return 0.1
    elif length < 200:
        return 0.5
    elif length < 1000:
        return 1.0
    elif length < 5000:
        return 1.5
    else:
        return 2.0


def type_priority_boost(node_type: str) -> float:
    """Get type priority multiplier."""
    if node_type in HIGH_VALUE_TYPES:
        return 3.0
    elif node_type in LOW_VALUE_TYPES:
        return 1.0
    elif node_type == "conversation_node":
        return 1.5  # Medium priority for substantial conversations
    else:
        return 1.0


def combined_node_score(vector_score: float, node_type: str, data: Dict[str, Any]) -> float:
    """
    Combined score = vector_score * type_boost * quality_boost.
    """
    quality = content_quality_score(data)
    
    if quality < 0:  # Tool output
        return vector_score * 0.01
    
    type_mult = type_priority_boost(node_type)
    quality_mult = 0.5 + quality / 2.0  # 0.5 to 1.5
    
    return vector_score * type_mult * quality_mult


class FAISSManager:
    """Manages FAISS index for semantic search."""
    
    def __init__(self, index_path: str = "brain_index.faiss", map_path: str = "brain_nodes_map.pkl"):
        self.index_path = Path(index_path)
        self.map_path = Path(map_path)
        self.index = None
        self.id_map = []  # list of node_ids
        self._load()
    
    def _load(self):
        if self.index_path.exists():
            self.index = get_faiss().read_index(str(self.index_path))
            with open(self.map_path, "rb") as f:
                self.id_map = pickle.load(f)
        else:
            self.index = get_faiss().IndexFlatIP(EMBED_DIM)
            self.id_map = []
    
    def save(self):
        get_faiss().write_index(self.index, str(self.index_path))
        with open(self.map_path, "wb") as f:
            pickle.dump(self.id_map, f)
    
    def add(self, node_id: str, embedding: np.ndarray):
        faiss_id = self.index.ntotal
        self.index.add(get_np().array(embedding).reshape(1, -1).astype(get_np().float32))
        self.id_map.append(node_id)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(
            get_np().array(query_embedding).reshape(1, -1).astype(get_np().float32), 
            top_k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.id_map):
                results.append((self.id_map[idx], float(score)))
        return results


def load_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def node_path(nodes_dir: Path, node_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(node_id))[:60] or "node"
    return nodes_dir / f"{safe}.json"


import re
import json


class BrainQuerier:
    """
    GraphRAG query layer for existing Hermes Brain.
    Quality-filtered vector search + graph expansion.
    """
    
    def __init__(
        self, 
        faiss_mgr: FAISSManager, 
        nodes_dir: Path,
        edges_dir: Path,
    ):
        self.faiss = faiss_mgr
        self.nodes_dir = Path(nodes_dir)
        self.edges_dir = Path(edges_dir)
        self._neighbor_cache = {}
    
    def query(
        self, 
        query_text: str, 
        top_k: int = 3, 
        expand_depth: int = 1,
        min_combined_score: float = 0.0
    ) -> str:
        """
        Execute GraphRAG query and return formatted context.
        """
        # 1. Embed query
        embedder = get_embedder()
        query_emb = embedder.encode(query_text)
        
        # 2. Vector search (get 3x candidates for filtering)
        candidates = self.faiss.search(query_emb, top_k * 3)
        
        # 3. Score and filter
        scored = []
        for node_id, vector_score in candidates:
            data = load_json(node_path(self.nodes_dir, node_id))
            if not data:
                continue
            
            node_type = data.get("type", "unknown")
            combined = combined_node_score(vector_score, node_type, data)
            
            if combined >= min_combined_score:
                scored.append((combined, node_id, vector_score, node_type, data))
        
        # 4. Sort by combined score
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        
        # 5. Format results with graph expansion
        if not top:
            return "No relevant context found in memory graph."
        
        return self._format_results(top, expand_depth)
    
    def _format_results(
        self, 
        top: List[Tuple], 
        expand_depth: int
    ) -> str:
        """Format query results as markdown context."""
        lines = ["## Relevant Context from Memory Graph\n"]
        
        for i, (combined, node_id, vector_score, node_type, data) in enumerate(top, 1):
            # Get main content
            content = ""
            if node_type == "conversation_node":
                content = data.get("properties", {}).get("content", "")
                role = data.get("properties", {}).get("role", "unknown")
                label = f"[{role}] {data.get('label', node_id)}"
            else:
                content = data.get("content", data.get("text", data.get("label", "")))
                label = data.get("label", node_id)
            
            # Truncate for display
            display_content = content[:2000] + ("..." if len(content) > 2000 else "")
            
            lines.append(f"### {i}. {label} ({node_type})")
            lines.append(f"**Score**: {combined:.3f} (vector: {vector_score:.3f})")
            lines.append(f"**Semantic Label**: {data.get('semantic_label', 'N/A')}")
            lines.append(f"**Cluster**: {data.get('community', 'N/A')}")
            lines.append(f"**Content**:\n{display_content}\n")
            
            # Graph expansion
            if expand_depth > 0:
                neighbors = self._get_neighbors(node_id, depth=expand_depth)
                if neighbors:
                    lines.append("**Related Nodes**:")
                    for n_id, n_type, n_label, rel in neighbors[:5]:
                        lines.append(f"  - {n_label} ({n_type}) [{rel}]")
                    lines.append("")
        
        return "\n".join(lines)
    
    def _get_neighbors(
        self, 
        node_id: str, 
        depth: int, 
        visited: Optional[set] = None
    ) -> List[Tuple[str, str, str, str]]:
        """Get neighboring nodes via edges."""
        if visited is None:
            visited = set()
        if node_id in visited or depth == 0:
            return []
        
        visited.add(node_id)
        neighbors = []
        
        # Find edges where this node is source or target
        for edge_file in self.edges_dir.glob("*.json"):
            edge = load_json(edge_file)
            if not edge:
                continue
            
            source = edge.get("source", "")
            target = edge.get("target", "")
            relation = edge.get("relation", "RELATES_TO")
            
            neighbor_id = None
            if source == node_id:
                neighbor_id = target
            elif target == node_id:
                neighbor_id = source
            
            if neighbor_id and neighbor_id not in visited:
                n_data = load_json(node_path(self.nodes_dir, neighbor_id))
                if n_data:
                    n_label = n_data.get("label", neighbor_id)
                    n_type = n_data.get("type", "unknown")
                    neighbors.append((neighbor_id, n_type, n_label, relation))
                    # Recurse
                    neighbors.extend(self._get_neighbors(neighbor_id, depth - 1, visited))
        
        return neighbors
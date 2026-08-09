"""
Evidence extraction and pattern matching utilities.
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ExtractedEntity:
    """An entity extracted from text."""
    text: str
    label: str  # PERSON, ORG, GPE, PRODUCT, TECH, CONCEPT, etc.
    confidence: float
    start: int
    end: int


class EntityExtractor:
    """Extracts entities using spaCy NER + custom regex patterns."""
    
    # Custom patterns for tech/business entities
    CUSTOM_PATTERNS = [
        (r'\b(?:DIAN|IVA|RUT|NIT|factura\s+electr[oó]nica)\b', 'TAX_CONCEPT', 0.9),
        (r'\b(?:Hermes|MCP|Brain|GraphRAG|FAISS|Louvain)\b', 'TECH_STACK', 0.9),
        (r'\b(?:Tecnosfera|Avena\s+Cubana|Pop)\b', 'BRAND', 0.9),
        (r'\b(?:Workana|Upwork|Freelancer|Fiverr)\b', 'PLATFORM', 0.8),
        (r'\b(?:Kimi\s*K?3|GLM-?5\.?2|Nemotron|DeepSeek|Qwen)\b', 'MODEL', 0.9),
        (r'\b(?:vLLM|SGLang|TokenSpeed|OpenInterpreter|Prime\s+Agent)\b', 'TOOL', 0.8),
        (r'\b(?:RLM|Continual\s+Harness|subagent|skill)\b', 'CONCEPT', 0.8),
        (r'\b\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s*(?:USD|COP|EUR)\b', 'MONEY', 0.9),
        (r'\b(?:https?://|www\.)\S+\b', 'URL', 0.95),
        (r'\b[A-Za-z]:[\\/][\w\\/.-]+\b', 'PATH', 0.9),
        (r'\b\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+)?\b', 'VERSION', 0.8),
        (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'IP', 0.9),
    ]
    
    def __init__(self):
        self._nlp = None
    
    @property
    def nlp(self):
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                # Fallback: blank model with only NER
                self._nlp = spacy.blank("en")
                # Add basic NER if available
        return self._nlp
    
    def extract(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from text using spaCy + custom patterns."""
        entities = []
        
        # spaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append(ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                confidence=0.85,
                start=ent.start_char,
                end=ent.end_char,
            ))
        
        # Custom regex patterns
        for pattern, label, conf in self.CUSTOM_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    label=label,
                    confidence=conf,
                    start=match.start(),
                    end=match.end(),
                ))
        
        # Deduplicate overlapping entities (keep highest confidence)
        entities = self._deduplicate(entities)
        return entities
    
    def _deduplicate(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove overlapping entities, keeping highest confidence."""
        if not entities:
            return []
        
        # Sort by start position, then by confidence descending
        entities.sort(key=lambda e: (e.start, -e.confidence))
        
        result = [entities[0]]
        for ent in entities[1:]:
            last = result[-1]
            # Check overlap
            if ent.start < last.end:
                # Overlapping - keep higher confidence
                if ent.confidence > last.confidence:
                    result[-1] = ent
            else:
                result.append(ent)
        
        return result


class PatternMatcher:
    """Matches behavioral patterns in conversation streams."""
    
    # Pattern definitions: (name, keywords, min_occurrences)
    PATTERNS = {
        "decision_making": (["decidí", "elegí", "opté", "concluí", "decided", "chose", "concluded"], 1),
        "error_correction": (["error", "bug", "falló", "fix", "corregí", "solucioné", "failed", "fixed"], 1),
        "preference_expression": (["prefiero", "me gusta", "no me gusta", "prefer", "like", "dislike"], 1),
        "tool_discovery": (["funciona", "no funciona", "mejor forma", "truco", "works", "doesn't work", "better way"], 2),
        "pattern_recognition": (["siempre que", "cuando", "noté que", "observé", "always when", "noticed", "pattern"], 2),
        "learning_moment": (["aprendí", "entendí", "ahora sé", "learned", "understood", "now I know"], 1),
        "workflow_optimization": (["más rápido", "optimiz", "automatiz", "faster", "optimize", "automat"], 1),
    }
    
    def __init__(self):
        # Compile regex for each pattern
        self.compiled = {}
        for name, (keywords, min_occ) in self.PATTERNS.items():
            # Create regex that matches any keyword
            kw_pattern = '|'.join(re.escape(kw) for kw in keywords)
            self.compiled[name] = (re.compile(kw_pattern, re.IGNORECASE), min_occ)
    
    def match(self, text: str) -> Dict[str, List[Dict]]:
        """Find all pattern matches in text."""
        matches = {}
        
        for name, (pattern, min_occ) in self.compiled.items():
            occurrences = []
            for match in pattern.finditer(text):
                occurrences.append({
                    "keyword": match.group(),
                    "position": match.start(),
                    "context": text[max(0, match.start()-50):match.end()+50],
                })
            
            if len(occurrences) >= min_occ:
                matches[name] = occurrences
        
        return matches
    
    def score_conversation(self, messages: List[Dict]) -> Dict[str, float]:
        """Score a conversation for learning potential."""
        scores = {name: 0.0 for name in self.PATTERNS}
        
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "")
            
            if role not in ("user", "assistant") or not content:
                continue
            
            matches = self.match(content)
            for name, occs in matches.items():
                # Weight by message role and length
                weight = 1.0 if role == "user" else 0.7
                weight *= min(len(content) / 500, 1.0)
                scores[name] += len(occs) * weight
        
        return scores


def extract_preferences(text: str) -> List[Dict[str, Any]]:
    """Extract user preferences from text."""
    prefs = []
    
    # Preference patterns
    patterns = [
        (r'prefiero\s+(.+?)(?:\.|$|,)', 'preference'),
        (r'me gusta\s+(.+?)(?:\.|$|,)', 'like'),
        (r'no me gusta\s+(.+?)(?:\.|$|,)', 'dislike'),
        (r'odio\s+(.+?)(?:\.|$|,)', 'hate'),
        (r'amo\s+(.+?)(?:\.|$|,)', 'love'),
        (r'I prefer\s+(.+?)(?:\.|$|,)', 'preference'),
        (r'I like\s+(.+?)(?:\.|$|,)', 'like'),
        (r'I don\'t like\s+(.+?)(?:\.|$|,)', 'dislike'),
        (r'I hate\s+(.+?)(?:\.|$|,)', 'hate'),
        (r'I love\s+(.+?)(?:\.|$|,)', 'love'),
    ]
    
    for pattern, pref_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            prefs.append({
                "type": pref_type,
                "value": match.group(1).strip(),
                "source_text": match.group(0),
                "confidence": 0.8,
            })
    
    return prefs


def extract_tool_usage(text: str) -> List[Dict[str, Any]]:
    """Extract tool usage patterns from text."""
    tools = []
    
    # Tool mention patterns
    patterns = [
        (r'(?:us[ée]|use|used)\s+(?:la\s+)?(?:herramienta\s+)?(\w+(?:[-_]\w+)*)', 'tool_used'),
        (r'(?:funciona|works)\s+(?:bien|good)?\s+(?:con|with)\s+(\w+(?:[-_]\w+)*)', 'tool_works'),
        (r'(?:no\s+)?(?:funciona|works?)\s+(?:con|with)\s+(\w+(?:[-_]\w+)*)', 'tool_issue'),
        (r'(?:mejor|better)\s+(?:forma|way)\s+(?:es|is)?\s+(\w+(?:[-_]\w+)*)', 'better_tool'),
    ]
    
    for pattern, usage_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            tools.append({
                "type": usage_type,
                "tool": match.group(1),
                "context": match.group(0),
                "confidence": 0.7,
            })
    
    return tools
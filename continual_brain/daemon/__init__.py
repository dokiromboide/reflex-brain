"""
Daemon Package - Background processing for Reflex Brain.
"""
from __future__ import annotations

from continual_brain.daemon.processor import run_daemon, DaemonState, SQLiteWatcher, EntityExtractor, NodeWriter, EdgeWriter, FAISSManager, BatchEmbedder, MessageProcessor, CommunityClassifier
from continual_brain.daemon.extractor import EntityExtractor as CoreEntityExtractor, PatternMatcher
from continual_brain.daemon.embedder import BatchEmbedder, FAISSManager

__all__ = [
    "run_daemon", "DaemonState", "SQLiteWatcher", "EntityExtractor", "NodeWriter", 
    "EdgeWriter", "FAISSManager", "BatchEmbedder", "MessageProcessor", "CommunityClassifier",
    "CoreEntityExtractor", "PatternMatcher",
]
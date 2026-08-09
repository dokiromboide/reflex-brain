"""
Daemon Package - Background processing for Reflex Brain.
"""
from __future__ import annotations

from continual_brain.daemon.embedder import BatchEmbedder, FAISSManager
from continual_brain.daemon.extractor import EntityExtractor as CoreEntityExtractor
from continual_brain.daemon.extractor import PatternMatcher
from continual_brain.daemon.processor import (
    BatchEmbedder,
    CommunityClassifier,
    DaemonState,
    EdgeWriter,
    EntityExtractor,
    FAISSManager,
    MessageProcessor,
    NodeWriter,
    SQLiteWatcher,
    run_daemon,
)

__all__ = [
    "run_daemon", "DaemonState", "SQLiteWatcher", "EntityExtractor", "NodeWriter",
    "EdgeWriter", "FAISSManager", "BatchEmbedder", "MessageProcessor", "CommunityClassifier",
    "CoreEntityExtractor", "PatternMatcher",
]

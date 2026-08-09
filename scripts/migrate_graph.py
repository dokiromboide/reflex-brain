#!/usr/bin/env python3
"""
Migration script: GraphRAG brain -> Continual Brain.
Usage: python scripts/migrate_graph.py --graph-dir brain --output continual.db
"""
import argparse
import asyncio
from pathlib import Path

from continual_brain.core.store import SQLiteStore
from continual_brain.core.migration import migrate_graphrag


async def main():
    parser = argparse.ArgumentParser(description="Migrate GraphRAG brain to Continual Brain")
    parser.add_argument("--graph-dir", required=True, help="GraphRAG brain directory (contains nodes/, edges/)")
    parser.add_argument("--output", default="continual.db", help="Output SQLite database path")
    parser.add_argument("--confidence", type=float, default=0.6, help="Base confidence for migrated lessons")
    args = parser.parse_args()
    
    graph_dir = Path(args.graph_dir)
    if not graph_dir.exists():
        print(f"Error: Graph directory not found: {graph_dir}")
        return 1
    
    print(f"Migrating GraphRAG brain from {graph_dir} to {args.output}...")
    print(f"Base confidence: {args.confidence}")
    
    store = SQLiteStore(args.output)
    await store.initialize()
    
    stats = migrate_graphrag(str(graph_dir), args.output, args.confidence)
    
    print("\n=== Migration Results ===")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    await store.close()
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
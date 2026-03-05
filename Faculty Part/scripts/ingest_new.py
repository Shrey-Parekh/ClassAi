#!/usr/bin/env python3
"""
New ingestion script using document-type-specific chunking and BAAI/bge-m3.

Usage:
    python scripts/ingest_new.py --input data/raw --metadata data/metadata.json
"""

import argparse
import traceback
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.ingestion.new_pipeline import NewIngestionPipeline
from src.utils.vector_db import VectorDBClient

# Load environment variables
load_dotenv()


def main():
    """Ingest documents using new chunking strategy with BAAI/bge-m3."""
    parser = argparse.ArgumentParser(
        description="Ingest faculty documents using new chunking strategy"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing documents to ingest"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="JSON file with document metadata"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="faculty_chunks",
        help="Vector DB collection name"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print(" "*20 + "FACULTY RAG INGESTION")
    print("="*70)
    print(f"\n📂 Input directory: {args.input}")
    print(f"📋 Metadata file:   {args.metadata if args.metadata else 'None'}")
    print(f"🗄️  Collection:      {args.collection}")
    print(f"🤖 Model:           BAAI/bge-m3 (1024 dimensions)")
    
    try:
        # Initialize vector DB
        print("\n" + "─"*70)
        print("STEP 1: Connecting to Qdrant")
        print("─"*70)
        vector_db = VectorDBClient(collection_name=args.collection)
        
        # Ensure collection exists with correct dimensions
        # bge-m3 uses 1024 dimensions
        try:
            info = vector_db.get_collection_info(args.collection)
            print(f"✓ Using existing collection: {args.collection}")
            print(f"  Current points: {info.get('points_count', 0)}")
        except:
            print(f"Creating new collection: {args.collection}")
            vector_db.create_collection(
                collection_name=args.collection,
                vector_size=1024  # bge-m3 dimensions
            )
            print(f"✓ Collection created: {args.collection}")
        
        # Initialize ingestion pipeline
        print("\n" + "─"*70)
        print("STEP 2: Loading Models")
        print("─"*70)
        pipeline = NewIngestionPipeline(
            vector_db_client=vector_db,
            collection_name=args.collection
        )
        
        # Ingest documents
        print("\n" + "─"*70)
        print("STEP 3: Processing Documents")
        print("─"*70)
        results = pipeline.ingest_directory(
            directory=args.input,
            metadata_file=args.metadata
        )
        
        # Check for failures
        failed = [r for r in results if "error" in r]
        if failed:
            print("\n" + "="*70)
            print("⚠ FAILED FILES")
            print("="*70)
            for r in failed:
                print(f"  ✗ {Path(r['file_path']).name}")
                print(f"     Error: {r['error'][:100]}")
        
        print("\n" + "="*70)
        print("✓ INGESTION COMPLETE")
        print("="*70)
        print(f"Collection:       {args.collection}")
        print(f"Files processed:  {len(results)}")
        print(f"Files succeeded:  {len(results) - len(failed)}")
        print(f"Files failed:     {len(failed)}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

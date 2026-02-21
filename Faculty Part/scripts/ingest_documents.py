#!/usr/bin/env python3
"""
Script to ingest faculty documents into the RAG system.

Usage:
    python scripts/ingest_documents.py --input data/raw --metadata data/metadata.json
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline
from src.utils.vector_db import VectorDBClient
from src.utils.embeddings import EmbeddingModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest faculty documents into RAG system"
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
    
    print(f"Ingesting documents from: {args.input}\n")
    
    try:
        # Initialize vector DB and embedding model
        print("Initializing components...")
        vector_db = VectorDBClient(collection_name=args.collection)
        embedding_model = EmbeddingModel()
        
        # Create collection if it doesn't exist
        vector_db.create_collection(
            collection_name=args.collection,
            vector_size=embedding_model.get_dimension()
        )
        
        # Initialize ingestion pipeline
        pipeline = IngestionPipeline(
            vector_db_client=vector_db,
            embedding_model=embedding_model,
            collection_name=args.collection
        )
        
        # Ingest documents
        print("\nIngesting documents...")
        results = pipeline.ingest_directory(
            directory=args.input,
            metadata_file=args.metadata
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print("Ingestion Complete!")
        print(f"{'='*60}")
        print(f"Total documents processed: {len(results)}")
        
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        total_chunks = sum(r.get('chunks_created', 0) for r in successful)
        print(f"Total chunks created: {total_chunks}")
        
        if failed:
            print("\nFailed documents:")
            for r in failed:
                print(f"  - {r['file_path']}: {r['error']}")
        
        # Show collection info
        print("\nCollection info:")
        info = vector_db.get_collection_info(args.collection)
        print(f"  Points: {info['points_count']}")
        print(f"  Status: {info['status']}")
        
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

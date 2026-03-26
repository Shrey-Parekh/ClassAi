#!/usr/bin/env python3
"""
Script to ingest faculty documents into the RAG system.

Usage:
    python scripts/ingest_documents.py --input data/raw --metadata data/metadata.json
"""

import argparse
import traceback
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from src.ingestion.pipeline import IngestionPipeline
from src.utils.vector_db import VectorDBClient
from src.utils.dual_encoder_embeddings import DualEncoderEmbeddings
from src.utils.llm import LLMClient

# Load environment variables
load_dotenv()


def main():
    """Ingest documents using BAAI/bge-large-en-v1.5 encoder."""
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
        # Initialize vector DB
        print("Initializing components...")
        vector_db = VectorDBClient(collection_name=args.collection)
        
        # Initialize encoder (BAAI/bge-m3)
        embedding_model = DualEncoderEmbeddings(
            model_name="BAAI/bge-m3",
            log_file="embedding_log.jsonl"
        )
        
        # Initialize LLM client for semantic chunking
        try:
            llm_client = LLMClient()
            print("✓ LLM client initialized for semantic chunking")
        except Exception as e:
            print(f"⚠ LLM client initialization failed (will use rule-based chunking): {e}")
            llm_client = None
        
        # Create collection with 1024 dimensions (BAAI/bge-m3)
        vector_db.create_collection(
            collection_name=args.collection,
            vector_size=1024
        )
        
        # Initialize ingestion pipeline
        pipeline = IngestionPipeline(
            vector_db_client=vector_db,
            embedding_model=embedding_model,
            collection_name=args.collection,
            llm_client=llm_client
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
        total_stored = sum(r.get('chunks_stored', 0) for r in successful)
        total_split = sum(r.get('chunks_split', 0) for r in successful)
        
        print(f"Total chunks created: {total_chunks}")
        print(f"Total chunks stored: {total_stored}")
        print(f"Total chunks split: {total_split}")
        
        if failed:
            print("\nFailed documents:")
            for r in failed:
                print(f"  - {r['file_path']}: {r['error']}")
        
        # Validate ingestion
        print("\nValidating ingestion...")
        try:
            collection_info = vector_db.get_collection_info(args.collection)
            actual_count = collection_info.get("points_count", 0)
            
            print(f"✓ Collection '{args.collection}' has {actual_count} points")
            
            if actual_count != total_stored:
                print(f"⚠ Warning: Expected {total_stored} chunks but found {actual_count} in collection")
            else:
                print(f"✓ Validation passed: All {total_stored} chunks stored successfully")
                
        except Exception as e:
            print(f"⚠ Could not validate collection: {e}")
        
    except Exception as e:
        print(f"\n✗ Ingestion failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to test queries against the RAG system.

Usage:
    python scripts/test_query.py "How do I apply for casual leave?"
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.pipeline import RetrievalPipeline
from src.generation.answer_generator import AnswerGenerator
from src.utils.vector_db import VectorDBClient
from src.utils.query_embedder import QueryEmbedder
from src.utils.llm import LLMClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Test query against Faculty Part RAG system"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Query to test"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="faculty_chunks",
        help="Vector DB collection name"
    )
    
    args = parser.parse_args()
    
    print(f"Query: {args.query}\n")
    print("Initializing RAG components...")
    
    try:
        # Initialize components
        vector_db = VectorDBClient(collection_name=args.collection)
        embedding_model = QueryEmbedder()
        llm_client = LLMClient()
        
        # Initialize retrieval pipeline
        retrieval_pipeline = RetrievalPipeline(
            vector_db_client=vector_db,
            embedding_model=embedding_model,
            collection_name=args.collection,
            llm_client=llm_client
        )
        
        # Build BM25 index for hybrid search
        print("Building BM25 index...")
        retrieval_pipeline.search_engine.build_bm25_index()
        
        # Initialize answer generator
        answer_generator = AnswerGenerator(llm_client)
        
        print("\nRetrieving relevant chunks...")
        retrieval_result = retrieval_pipeline.retrieve(
            query=args.query,
            top_k=args.top_k
        )
        
        print(f"Intent detected: {retrieval_result['intent']}")
        print(f"Chunks retrieved: {len(retrieval_result['chunks'])}\n")
        
        print("Generating answer...")
        answer_result = answer_generator.generate(
            query=args.query,
            retrieved_chunks=retrieval_result["chunks"],
            intent_type=retrieval_result["intent"]
        )
        
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(answer_result["answer"])
        
        print("\n" + "="*60)
        print("SOURCES:")
        print("="*60)
        for source in answer_result["sources"]:
            print(f"  • {source['title']}")
            if source.get('date'):
                print(f"    Date: {source['date']}")
            if source.get('applies_to'):
                print(f"    Applies to: {source['applies_to']}")
        
        print("\n" + "="*60)
        print("METADATA:")
        print("="*60)
        print(f"  Intent: {retrieval_result['intent']}")
        print(f"  Chunks used: {answer_result['chunks_used']}")
        print(f"  Initial results: {retrieval_result['metadata']['initial_results']}")
        
    except Exception as e:
        print(f"\n✗ Query failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

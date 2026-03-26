"""
Evaluation baseline for RAG system.

Run this before and after changes to measure impact.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import retrieval_pipeline, answer_generator


# Test cases format:
# {
#     "question": "Who is Dr. Pragati Khare?",
#     "expected_answer_snippet": "Associate Professor",
#     "expected_source_doc": "facult_data.json",
#     "category": "faculty_lookup"
# }

TEST_CASES = [
    # Faculty lookup cases
    {
        "question": "Who is Dr. Pragati Khare?",
        "expected_answer_snippet": "computer science",
        "expected_source_doc": "facult_data.json",
        "category": "faculty_lookup"
    },
    {
        "question": "What is Pragati Khare's research area?",
        "expected_answer_snippet": "machine learning",
        "expected_source_doc": "facult_data.json",
        "category": "faculty_research"
    },
    
    # Policy cases
    {
        "question": "What is the casual leave policy?",
        "expected_answer_snippet": "12 days",
        "expected_source_doc": "NMIMS_Employee_Resource_Book",
        "category": "policy_lookup"
    },
    {
        "question": "How do I apply for leave?",
        "expected_answer_snippet": "form",
        "expected_source_doc": "NMIMS_Employee_Resource_Book",
        "category": "procedure"
    },
    
    # Eligibility cases
    {
        "question": "Am I eligible for sabbatical leave?",
        "expected_answer_snippet": "years of service",
        "expected_source_doc": "NMIMS_Employee_Resource_Book",
        "category": "eligibility"
    },
    
    # Form cases
    {
        "question": "What forms do I need for reimbursement?",
        "expected_answer_snippet": "reimbursement",
        "expected_source_doc": "NMIMS_Faculty_Applications_Compendium",
        "category": "form_lookup"
    },
    
    # General cases
    {
        "question": "What are the faculty guidelines?",
        "expected_answer_snippet": "guideline",
        "expected_source_doc": "NMIMS_Faculty_Academic_Guidelines",
        "category": "general"
    },
]


def run_evaluation(test_cases: List[Dict[str, Any]], output_file: str = None) -> Dict[str, Any]:
    """
    Run evaluation on test cases.
    
    Args:
        test_cases: List of test case dicts
        output_file: Optional file to save results
    
    Returns:
        Evaluation results
    """
    if not retrieval_pipeline or not answer_generator:
        print("ERROR: RAG system not initialized. Start the API server first.")
        return {}
    
    results = []
    passed = 0
    failed = 0
    
    print(f"\n{'='*60}")
    print(f"Running evaluation on {len(test_cases)} test cases...")
    print(f"{'='*60}\n")
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected_snippet = test_case["expected_answer_snippet"].lower()
        expected_doc = test_case["expected_source_doc"].lower()
        category = test_case["category"]
        
        print(f"[{i}/{len(test_cases)}] {category}: {question}")
        
        try:
            # Retrieve
            retrieval_result = retrieval_pipeline.retrieve(question, top_k=15)
            
            # Generate
            answer_result = answer_generator.generate(
                query=question,
                retrieved_chunks=retrieval_result["chunks"],
                intent_type=retrieval_result["intent"]
            )
            
            # Check answer
            answer_text = json.dumps(answer_result["structured"]).lower()
            answer_contains_snippet = expected_snippet in answer_text
            
            # Check sources
            sources = answer_result["sources"]
            source_docs = [s["title"].lower() for s in sources]
            correct_source = any(expected_doc in doc for doc in source_docs)
            
            # Pass if both conditions met
            passed_test = answer_contains_snippet and correct_source
            
            if passed_test:
                passed += 1
                status = "✓ PASS"
            else:
                failed += 1
                status = "✗ FAIL"
                if not answer_contains_snippet:
                    status += f" (missing '{expected_snippet}')"
                if not correct_source:
                    status += f" (wrong source, expected '{expected_doc}')"
            
            print(f"  {status}")
            print(f"  Intent: {retrieval_result['intent']}, Chunks: {len(retrieval_result['chunks'])}")
            
            results.append({
                "question": question,
                "category": category,
                "passed": passed_test,
                "answer_contains_snippet": answer_contains_snippet,
                "correct_source": correct_source,
                "intent": retrieval_result["intent"],
                "chunks_retrieved": len(retrieval_result["chunks"]),
                "sources": source_docs,
                "answer_preview": answer_text[:200]
            })
            
        except Exception as e:
            failed += 1
            print(f"  ✗ ERROR: {str(e)[:100]}")
            results.append({
                "question": question,
                "category": category,
                "passed": False,
                "error": str(e)
            })
        
        print()
    
    # Summary
    total = len(test_cases)
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "results": results
    }
    
    print(f"{'='*60}")
    print(f"SUMMARY: {passed}/{total} passed ({pass_rate:.1f}%)")
    print(f"{'='*60}\n")
    
    # Save to file
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to: {output_file}\n")
    
    return summary


if __name__ == "__main__":
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tests/eval_results/baseline_{timestamp}.json"
    
    run_evaluation(TEST_CASES, output_file)

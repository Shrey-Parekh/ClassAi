"""
Analyze context window usage from logs.

Run after collecting usage data to identify if limits are too loose/tight.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import statistics


def analyze_context_usage(log_file: str = "logs/context_usage.jsonl") -> Dict:
    """
    Analyze context usage logs.
    
    Args:
        log_file: Path to context usage log file
    
    Returns:
        Analysis results
    """
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_file}")
        print("Run queries first to generate usage data.")
        return {}
    
    # Read all log entries
    entries = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not entries:
        print("No valid log entries found.")
        return {}
    
    # Extract metrics
    tokens_used = [e["tokens_used"] for e in entries]
    chunks_used = [e["chunks_used"] for e in entries]
    utilization = [e["utilization"] for e in entries]
    
    # Calculate statistics
    analysis = {
        "total_queries": len(entries),
        "tokens_used": {
            "min": min(tokens_used),
            "max": max(tokens_used),
            "mean": statistics.mean(tokens_used),
            "median": statistics.median(tokens_used),
            "p90": statistics.quantiles(tokens_used, n=10)[8] if len(tokens_used) > 10 else max(tokens_used),
            "p99": statistics.quantiles(tokens_used, n=100)[98] if len(tokens_used) > 100 else max(tokens_used)
        },
        "chunks_used": {
            "min": min(chunks_used),
            "max": max(chunks_used),
            "mean": statistics.mean(chunks_used),
            "median": statistics.median(chunks_used)
        },
        "utilization": {
            "min": min(utilization),
            "max": max(utilization),
            "mean": statistics.mean(utilization),
            "median": statistics.median(utilization),
            "p90": statistics.quantiles(utilization, n=10)[8] if len(utilization) > 10 else max(utilization)
        }
    }
    
    # Print report
    print("\n" + "="*60)
    print("CONTEXT WINDOW USAGE ANALYSIS")
    print("="*60)
    print(f"\nTotal Queries Analyzed: {analysis['total_queries']}")
    
    print("\n--- Tokens Used ---")
    print(f"  Min:    {analysis['tokens_used']['min']:>6.0f}")
    print(f"  Median: {analysis['tokens_used']['median']:>6.0f}")
    print(f"  Mean:   {analysis['tokens_used']['mean']:>6.0f}")
    print(f"  P90:    {analysis['tokens_used']['p90']:>6.0f}")
    print(f"  Max:    {analysis['tokens_used']['max']:>6.0f}")
    
    print("\n--- Chunks Used ---")
    print(f"  Min:    {analysis['chunks_used']['min']:>6.1f}")
    print(f"  Median: {analysis['chunks_used']['median']:>6.1f}")
    print(f"  Mean:   {analysis['chunks_used']['mean']:>6.1f}")
    print(f"  Max:    {analysis['chunks_used']['max']:>6.1f}")
    
    print("\n--- Utilization (% of available) ---")
    print(f"  Min:    {analysis['utilization']['min']*100:>6.1f}%")
    print(f"  Median: {analysis['utilization']['median']*100:>6.1f}%")
    print(f"  Mean:   {analysis['utilization']['mean']*100:>6.1f}%")
    print(f"  P90:    {analysis['utilization']['p90']*100:>6.1f}%")
    print(f"  Max:    {analysis['utilization']['max']*100:>6.1f}%")
    
    # Recommendations
    print("\n--- Recommendations ---")
    mean_util = analysis['utilization']['mean']
    p90_util = analysis['utilization']['p90']
    
    if p90_util > 0.9:
        print("  ⚠ HIGH UTILIZATION: P90 > 90%")
        print("  → Chunk limits are too loose")
        print("  → Reduce INTENT_CHUNK_LIMITS by 20-30%")
        print("  → Risk of context overflow")
    elif mean_util < 0.3:
        print("  ℹ LOW UTILIZATION: Mean < 30%")
        print("  → Chunk limits are conservative")
        print("  → Can increase INTENT_CHUNK_LIMITS by 30-50%")
        print("  → Opportunity to improve recall")
    else:
        print("  ✓ OPTIMAL UTILIZATION: 30-90%")
        print("  → Current limits are well-tuned")
        print("  → No changes needed")
    
    print("\n" + "="*60 + "\n")
    
    return analysis


if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/context_usage.jsonl"
    analyze_context_usage(log_file)

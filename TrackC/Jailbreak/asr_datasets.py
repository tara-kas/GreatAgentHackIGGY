import os
import csv
import json
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Import HLE analysis functions from hle_benchmark_runner
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hle_benchmark_runner import make_api_call, detect_refusal, AGENTS

# Try to import datasets for HarmBench
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:
    _load_dotenv = None

try:
    from huggingface_hub import login as _hf_login
except Exception:
    _hf_login = None

# Set up output directory for HB results
ROOT_DIR = Path(__file__).parent.parent.parent
HB_RESULTS_DIR = ROOT_DIR / "TrackC" / "Jailbreak" / "HB_benchmark"
HB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_hf_auth() -> None:
    """Ensure HuggingFace authentication is set up."""
    if _load_dotenv is not None:
        _load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if token and _hf_login is not None:
        try:
            _hf_login(token=token, add_to_git_credential=False)
        except Exception:
            pass


def load_harmbench_samples(n: int | None = None, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load samples from HarmBench dataset.
    
    Args:
        n: Number of samples to load (None for all)
        seed: Random seed for sampling
    
    Returns:
        List of sample dictionaries with prompt, category, context, etc.
    """
    if load_dataset is None:
        raise RuntimeError("datasets is not installed. pip install datasets")
    
    _ensure_hf_auth()
    
    # Load HarmBench contextual variant
    ds = load_dataset("walledai/HarmBench", "standard", split="train")
    cols = list(ds.column_names)
    
    has_context = 'context' in cols
    category_field = 'category' if 'category' in cols else None
    
    # Find the prompt field
    prompt_field = None
    for c in ["prompt", "question", "attack", "input", "text"]:
        if c in cols:
            prompt_field = c
            break
    if prompt_field is None:
        for c in cols:
            if isinstance(ds[0][c], str):
                prompt_field = c
                break
    if prompt_field is None:
        raise RuntimeError(f"Could not find a text field in HarmBench columns: {cols}")
    
    # Process all samples by default, or sample if n is specified and less than total
    total_samples = len(ds)
    if n is not None and n < total_samples:
        # Sample n items randomly
        rnd = random.Random(seed)
        indices = list(range(total_samples))
        rnd.shuffle(indices)
        indices = indices[:n]
        print(f"  Sampling {n} out of {total_samples} total samples (seed={seed})")
    else:
        # Process all samples
        indices = list(range(total_samples))
        print(f"  Processing all {total_samples} samples from HarmBench dataset")
    
    items: List[Dict[str, Any]] = []
    for i in indices:
        row = ds[i]
        p = row.get(prompt_field)
        ctx = row.get('context') if has_context else None
        
        if not isinstance(p, str):
            p = str(p)
        
        # Combine context and prompt if context exists
        if isinstance(ctx, str) and ctx.strip():
            combined = f"{ctx.strip()}\n\n{p.strip()}"
        else:
            combined = p.strip()
        
        items.append({
            "index": i,
            "prompt": combined,
            "context": ctx if isinstance(ctx, str) else None,
            "raw_prompt": p,
            "category": row.get(category_field) if category_field else None,
        })
    
    return items


def process_single_request(agent: str, question: str, prompt_idx: int) -> Tuple[int, str, Dict[str, Any]]:
    """
    Process a single (agent, prompt) request.
    
    Returns:
        Tuple of (prompt_idx, agent, result_dict)
    """
    result = make_api_call(
        agent,
        question,
        None,  # No shared session - each request creates its own connection
        1,            # max_retries
        0.5,          # initial_wait
        (5, 15),      # connect/read timeout
    )
    return (prompt_idx, agent, result)


def process_harmbench_samples(samples: List[Dict[str, Any]], 
                            agents: List[str] | None = None, max_workers: int = 100) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str], List[Dict[str, Any]]]:
    """
    Process HarmBench samples in parallel - ALL (prompt, agent) combinations processed simultaneously.
    
    Args:
        samples: List of sample dictionaries from HarmBench
        agents: List of agents to test (defaults to all AGENTS)
        max_workers: Maximum number of parallel workers (all requests sent simultaneously)
    
    Returns:
        Tuple of (main_rows, diag_rows, main_fieldnames, diag_fieldnames, jsonl_entries) for CSV and JSONL writing
    """
    if agents is None:
        agents = AGENTS
    
    # Prepare CSV fieldnames
    main_fieldnames = ["question"]
    for a in agents:
        main_fieldnames.append(a)
        main_fieldnames.append(f"{a}_response")
    
    diag_fieldnames = ["question"] + [f"{a}_status" for a in agents] + [f"{a}_latency_ms" for a in agents]
    
    # Prepare all prompts with indices and categories
    prompts = [(idx, sample["prompt"], sample.get("category")) 
               for idx, sample in enumerate(samples)]
    
    total_requests = len(prompts) * len(agents)
    print(f"\nProcessing HarmBench samples ({len(prompts)} prompts × {len(agents)} agents = {total_requests} requests)...")
    print(f"  Sending ALL {total_requests} requests in parallel (max_workers={max_workers})...")
    
    # Initialize result structures
    main_rows: List[Dict[str, Any]] = [{"question": question} for _, question, _ in prompts]
    diag_rows: List[Dict[str, Any]] = [{"question": question} for _, question, _ in prompts]
    jsonl_entries: List[Dict[str, Any]] = []
    
    # Initialize all agent fields
    for row_success, diag_row in zip(main_rows, diag_rows):
        for agent in agents:
            row_success[agent] = "0"
            row_success[f"{agent}_response"] = ""
            diag_row[f"{agent}_status"] = ""
            diag_row[f"{agent}_latency_ms"] = ""
    
    # Submit ALL (prompt, agent) combinations in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, total_requests)) as ex:
        # Create all futures
        futures = {
            ex.submit(process_single_request, agent, question, prompt_idx): (prompt_idx, agent, question, category)
            for prompt_idx, question, category in prompts
            for agent in agents
        }
        
        # Process results as they complete
        for fut in as_completed(futures):
            prompt_idx, agent, question, category = futures[fut]
            try:
                _, _, result = fut.result()
                
                status = result.get("status_code")
                text = result.get("text")
                latency = result.get("latency_ms")
                error = result.get("error")
                
                # Check if this is a timeout - skip saving to JSONL if so
                is_timeout = error is not None and "timeout" in str(error).lower()
                
                refused = detect_refusal(text or "") if error is None and status == 200 else True
                success = 0 if refused else 1
                
                main_rows[prompt_idx][agent] = str(success)
                main_rows[prompt_idx][f"{agent}_response"] = (text or "")[:500]
                diag_rows[prompt_idx][f"{agent}_status"] = str(status) if status is not None else ""
                diag_rows[prompt_idx][f"{agent}_latency_ms"] = str(latency) if latency is not None else ""
                
                # Create JSONL entry with actual category from dataset (skip if timeout)
                if not is_timeout:
                    jsonl_entry = {
                        "prompt": question,
                        "category": category if category else "unknown",
                        "agent": agent,
                        "status": status,
                        "latency_ms": latency,
                        "error": error,
                        "response": text or "",
                        "judgments": {
                            "heuristic": 1 if not refused else 0,
                            "rules": 0  # Rules-based judgment not implemented yet
                        }
                    }
                    jsonl_entries.append(jsonl_entry)
                
                completed += 1
                if completed % 50 == 0 or completed == total_requests:
                    print(f"  Completed {completed}/{total_requests} requests...")
                    
            except Exception as e:
                print(f"  Error processing (prompt {prompt_idx}, agent {agent}): {e}")
                # Check if this is a timeout - skip saving to JSONL if so
                is_timeout = "timeout" in str(e).lower()
                if not is_timeout:
                    # Still create JSONL entry for non-timeout errors
                    jsonl_entry = {
                        "prompt": question,
                        "category": category if category else "unknown",
                        "agent": agent,
                        "status": None,
                        "latency_ms": None,
                        "error": str(e),
                        "response": "",
                        "judgments": {
                            "heuristic": 0,
                            "rules": 0
                        }
                    }
                    jsonl_entries.append(jsonl_entry)
    
    print(f"  Completed all {total_requests} requests!")
    
    return main_rows, diag_rows, main_fieldnames, diag_fieldnames, jsonl_entries


def write_results(main_rows: List[Dict[str, Any]], diag_rows: List[Dict[str, Any]], 
                 main_fieldnames: List[str], diag_fieldnames: List[str],
                 jsonl_entries: List[Dict[str, Any]],
                 agents: List[str] | None = None):
    """
    Write results to CSV files and JSONL file, and compute ASR.
    """
    if agents is None:
        agents = AGENTS
    
    # Compute ASR per agent
    totals = {a: 0 for a in agents}
    for r in main_rows:
        for a in agents:
            totals[a] += int(r.get(a, "0") or "0")
    
    asr_row: Dict[str, Any] = {fn: "" for fn in main_fieldnames}
    asr_row["question"] = "ASR"
    denom = max(1, len(main_rows))
    for a in agents:
        asr_row[a] = totals[a] / denom
    
    # Write main CSV
    main_csv_path = HB_RESULTS_DIR / "hb_results_main.csv"
    with open(main_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=main_fieldnames)
        writer.writeheader()
        for r in main_rows:
            writer.writerow(r)
        writer.writerow(asr_row)
    print(f"  Written main results to {main_csv_path}")
    
    # Write diagnostics CSV
    diag_csv_path = HB_RESULTS_DIR / "hb_results_diagnostics.csv"
    with open(diag_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=diag_fieldnames)
        writer.writeheader()
        for r in diag_rows:
            writer.writerow(r)
    print(f"  Written diagnostics to {diag_csv_path}")
    
    # Write JSONL file
    jsonl_path = HB_RESULTS_DIR / "hb_results_attempts.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Written JSONL attempts to {jsonl_path}")
    
    # Print ASR summary
    print(f"\n  ASR (naive heuristic; 1=success, 0=refusal):")
    for a in agents:
        asr = totals[a] / max(1, len(main_rows))
        print(f"    - {a}: {asr:.2f}")


def main(n_samples: int | None = None, seed: int = 42):
    """
    Run HarmBench evaluation.
    
    Args:
        n_samples: Number of samples to process (None for all)
        seed: Random seed for sampling
    """
    print("="*60)
    print("HarmBench Dataset Evaluation")
    print("="*60)
    
    # Load HarmBench samples (all by default)
    if n_samples is None:
        print(f"\nLoading all HarmBench samples from dataset...")
    else:
        print(f"\nLoading HarmBench samples (n={n_samples}, seed={seed})...")
    samples = load_harmbench_samples(n=n_samples, seed=seed)
    print(f"  Loaded {len(samples)} samples")
    
    # Process all samples
    main_rows, diag_rows, main_fieldnames, diag_fieldnames, jsonl_entries = process_harmbench_samples(
        samples
    )
    
    # Write results
    write_results(main_rows, diag_rows, main_fieldnames, diag_fieldnames, jsonl_entries)
    
    print("\n" + "="*60)
    print("All tests completed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run HarmBench evaluation")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all samples in dataset)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    args = parser.parse_args()
    main(n_samples=args.n_samples, seed=args.seed)
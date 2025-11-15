import os
import csv
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Import HLE analysis functions from hle_benchmark_runner
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hle_benchmark_runner import make_api_call, detect_refusal, AGENTS

path = os.path.join(os.path.dirname(__file__), "../RedTeamingPrompts")

# Set up output directory for HLE results
ROOT_DIR = Path(__file__).parent.parent.parent
HLE_RESULTS_DIR = ROOT_DIR / "TrackC" / "Jailbreak" / "hle-results"
HLE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

benign_df = pd.read_csv(os.path.join(path, "benign_test_cases.csv"))
harmful_df = pd.read_csv(os.path.join(path, "harmful_test_cases.csv"))
jailbreak_df = pd.read_csv(os.path.join(path, "jailbreak_prompts.csv"))


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


def process_prompt_category(df: pd.DataFrame, category_name: str, question_column: str = "question", 
                            agents: List[str] | None = None, max_workers: int = 200) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
    """
    Process a category of prompts in parallel - ALL (prompt, agent) combinations processed simultaneously.
    
    Args:
        df: DataFrame containing prompts
        category_name: Name of the category (for output files)
        question_column: Column name containing the prompt/question
        agents: List of agents to test (defaults to all AGENTS)
        max_workers: Maximum number of parallel workers (all requests sent simultaneously)
    
    Returns:
        Tuple of (main_rows, diag_rows, main_fieldnames, diag_fieldnames) for CSV writing
    """
    if agents is None:
        agents = AGENTS
    
    # Prepare CSV fieldnames
    main_fieldnames = ["question"]
    for a in agents:
        main_fieldnames.append(a)
        main_fieldnames.append(f"{a}_response")
    
    diag_fieldnames = ["question"] + [f"{a}_status" for a in agents] + [f"{a}_latency_ms" for a in agents]
    
    # Prepare all prompts with indices
    prompts = [(idx, str(getattr(row, question_column))) 
               for idx, row in enumerate(df.itertuples(index=False))]
    
    total_requests = len(prompts) * len(agents)
    print(f"\nProcessing {category_name} queries ({len(prompts)} prompts × {len(agents)} agents = {total_requests} requests)...")
    print(f"  Sending ALL {total_requests} requests in parallel (max_workers={max_workers})...")
    
    # Initialize result structures
    main_rows: List[Dict[str, Any]] = [{"question": question} for _, question in prompts]
    diag_rows: List[Dict[str, Any]] = [{"question": question} for _, question in prompts]
    
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
            ex.submit(process_single_request, agent, question, prompt_idx): (prompt_idx, agent)
            for prompt_idx, question in prompts
            for agent in agents
        }
        
        # Process results as they complete
        for fut in as_completed(futures):
            prompt_idx, agent = futures[fut]
            try:
                _, _, result = fut.result()
                
                status = result.get("status_code")
                text = result.get("text")
                latency = result.get("latency_ms")
                error = result.get("error")
                
                refused = detect_refusal(text or "") if error is None and status == 200 else True
                success = 0 if refused else 1
                
                main_rows[prompt_idx][agent] = str(success)
                main_rows[prompt_idx][f"{agent}_response"] = (text or "")[:500]
                diag_rows[prompt_idx][f"{agent}_status"] = str(status) if status is not None else ""
                diag_rows[prompt_idx][f"{agent}_latency_ms"] = str(latency) if latency is not None else ""
                
                completed += 1
                if completed % 50 == 0 or completed == total_requests:
                    print(f"  Completed {completed}/{total_requests} requests...")
                    
            except Exception as e:
                print(f"  Error processing (prompt {prompt_idx}, agent {agent}): {e}")
    
    print(f"  Completed all {total_requests} requests for {category_name}!")
    
    return main_rows, diag_rows, main_fieldnames, diag_fieldnames


def write_results(main_rows: List[Dict[str, Any]], diag_rows: List[Dict[str, Any]], 
                 main_fieldnames: List[str], diag_fieldnames: List[str],
                 category_name: str, agents: List[str] | None = None):
    """
    Write results to CSV files and compute ASR.
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
    main_csv_path = HLE_RESULTS_DIR / f"hle_results_{category_name}_main.csv"
    with open(main_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=main_fieldnames)
        writer.writeheader()
        for r in main_rows:
            writer.writerow(r)
        writer.writerow(asr_row)
    print(f"  Written main results to {main_csv_path}")
    
    # Write diagnostics CSV
    diag_csv_path = HLE_RESULTS_DIR / f"hle_results_{category_name}_diagnostics.csv"
    with open(diag_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=diag_fieldnames)
        writer.writeheader()
        for r in diag_rows:
            writer.writerow(r)
    print(f"  Written diagnostics to {diag_csv_path}")
    
    # Print ASR summary
    print(f"\n  ASR for {category_name} (naive heuristic; 1=success, 0=refusal):")
    for a in agents:
        asr = totals[a] / max(1, len(main_rows))
        print(f"    - {a}: {asr:.2f}")


def main():
    """Run all test categories in parallel."""
    # Test with benign queries
    benign_main, benign_diag, benign_main_fields, benign_diag_fields = process_prompt_category(
        benign_df, "benign", question_column="question"
    )
    write_results(benign_main, benign_diag, benign_main_fields, benign_diag_fields, "benign")
    
    # Test with harmful queries
    harmful_main, harmful_diag, harmful_main_fields, harmful_diag_fields = process_prompt_category(
        harmful_df, "harmful", question_column="question"
    )
    write_results(harmful_main, harmful_diag, harmful_main_fields, harmful_diag_fields, "harmful")
    
    # Test with jailbreak prompts
    jailbreak_main, jailbreak_diag, jailbreak_main_fields, jailbreak_diag_fields = process_prompt_category(
        jailbreak_df, "jailbreak", question_column="prompt"
    )
    write_results(jailbreak_main, jailbreak_diag, jailbreak_main_fields, jailbreak_diag_fields, "jailbreak")
    
    print("\n" + "="*60)
    print("All tests completed!")


if __name__ == "__main__":
    main()
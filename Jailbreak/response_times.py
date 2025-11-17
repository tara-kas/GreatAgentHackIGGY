import json
from collections import defaultdict
from typing import Dict, Iterable
from pathlib import Path


def load_hb_attempts(path: str) -> Iterable[dict]:
    """
    Stream JSONL results from `hb_results_attempts.jsonl`.

    Each line is expected to be a JSON object with at least:
    - 'agent': str
    - 'latency_ms': number (int or float)
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines rather than failing the whole run
                continue


def average_response_times_by_agent(path: str) -> Dict[str, float]:
    """
    Compute the average response time (in ms) for each agent
    from an `hb_results_attempts.jsonl` file.

    Parameters
    ----------
    path : str
        Path to the `hb_results_attempts.jsonl` file.

    Returns
    -------
    Dict[str, float]
        Mapping from agent name to average latency in milliseconds.
    """
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)

    for rec in load_hb_attempts(path):
        agent = rec.get("agent")
        latency = rec.get("latency_ms")

        if agent is None or latency is None:
            continue

        try:
            latency_val = float(latency)
        except (TypeError, ValueError):
            continue

        totals[agent] += latency_val
        counts[agent] += 1

    return {
        agent: (totals[agent] / counts[agent])
        for agent in totals
        if counts[agent] > 0
    }


if __name__ == "__main__":
    # Example usage:
    # python response_times.py
    # (adjust the path below if needed)
    path = Path(__file__).parent.parent.parent / "TrackC" / "Jailbreak" / "HB_benchmark" / "hb_results_attempts.jsonl"
    averages = average_response_times_by_agent(str(path))
    for agent, avg in sorted(averages.items(), key=lambda x: x[0]):
        print(f"{agent}: {avg:.2f} ms")



import json
import sys
from typing import Iterable, Mapping, Any
from .util import judge_attack

def _normalize_attempt(record: Mapping[str, Any]) -> Mapping[str, Any]:
    original = record.get("original_prompt") or record.get("original") or record.get("question") or record.get("prompt") or ""
    attack = record.get("attack_prompt") or record.get("attack") or record.get("jailbreak") or record.get("prompt_override") or record.get("prompt") or ""
    response = record.get("response") or record.get("answer") or record.get("model_response") or record.get("message") or record.get("output")
    return {"original_prompt": original, "attack_prompt": attack, "response": response}

def _judge(record: Mapping[str, Any]) -> bool:
    r = _normalize_attempt(record)
    result = judge_attack(r["original_prompt"], r["attack_prompt"], r["response"]) or {}
    return bool(result.get("attack_succeeded", False))

def calculate_asr(attempts: Iterable[Mapping[str, Any]]) -> float:
    total = 0
    succeeded = 0
    for rec in attempts:
        if not isinstance(rec, Mapping):
            continue
        norm = _normalize_attempt(rec)
        if not (norm["attack_prompt"] or norm["original_prompt"]) or norm.get("response") is None:
            continue
        total += 1
        try:
            if _judge(norm):
                succeeded += 1
        except Exception:
            pass
    if total == 0:
        return 0.0
    return succeeded / total

def _iter_jsonl(path: str) -> Iterable[Mapping[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m TrackC.Jailbreak.calculate_asr <attempts.jsonl>")
        sys.exit(2)
    attempts_path = sys.argv[1]
    asr = calculate_asr(_iter_jsonl(attempts_path))
    print(f"{asr:.6f}")


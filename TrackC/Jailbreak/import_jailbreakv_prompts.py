import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from datasets import load_dataset


Category = Literal[
    "DAN_attack",
    "role_play",
    "sti_attack",
    "please_attack",
    "function_attacks",
]


BASE_DIR = Path(__file__).resolve().parents[1] / "RedTeamingPrompts"


def load_existing(file_path: Path) -> Tuple[List[Dict], str]:
    """
    Load existing prompts from a file that may be either:
    - a JSON array, or
    - JSONL (one JSON object per line)

    Returns (items, style) where style is "array" or "jsonl".
    """
    if not file_path.exists():
        return [], "jsonl"

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return [], "jsonl"

    # Detect array vs JSONL by first non-whitespace character
    first_char = text[0]
    if first_char == "[":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data, "array"
        except json.JSONDecodeError:
            pass

    # Fallback to JSONL
    items: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            # Skip malformed lines rather than failing the whole script
            continue
    return items, "jsonl"


def save_items(file_path: Path, items: List[Dict], style: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if style == "array":
        file_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        with file_path.open("w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def classify_prompt(jailbreak_query: str, fmt: str) -> Category | None:
    jq = jailbreak_query or ""
    jq_lower = jq.lower()
    fmt_lower = (fmt or "").lower()

    # 1. Explicit DAN-style attacks
    if " dan," in jq_lower or " dan " in jq or "do anything now" in jq_lower or "[8.6-rc1:" in jq_lower:
        return "DAN_attack"

    # 2. Function / tool-call style attacks
    if any(tok in jq for tok in ["<|tool_call|>", "<|python_tag|>", "<|bash_tag|>"]) or (
        "{\"name\"" in jq and "\"parameters\"" in jq
    ):
        return "function_attacks"

    # 3. System / prompt-injection style (sti-style) attacks
    if any(
        phrase in jq_lower
        for phrase in [
            "<|im_start|>system",
            "ignore all previous instructions",
            "ignore previous instructions",
            "override the system prompt",
            "system prompt",
            "[system override]",
        ]
    ):
        return "sti_attack"

    # 4. Role-play style jailbreaks
    if any(
        phrase in jq_lower
        for phrase in [
            "let's role play",
            "role-play as",
            "role play as",
            "pretend to be",
            "act as ",
            "you will now play the role",
            "from now on, you are",
        ]
    ):
        return "role_play"

    # 5. "Please" persuasive attacks
    if "please" in jq_lower or "i beg you" in jq_lower or "i'm begging you" in jq_lower:
        return "please_attack"

    # 6. Fallback: use the high-level format from JailBreakV-28K
    # Template / Persuade / Logic / figstep / query-relevant
    if fmt_lower == "template":
        return "DAN_attack"
    if fmt_lower == "persuade":
        return "please_attack"
    if fmt_lower == "logic":
        return "sti_attack"
    if fmt_lower in {"figstep", "query-relevant"}:
        return "role_play"

    return None


def main() -> None:
    # Load full JailBreakV-28K jailbreak split
    # Reference: https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k
    ds = load_dataset("JailbreakV-28K/JailBreakV-28k", "JailBreakV_28K")["JailBreakV_28K"]

    target_per_category: Dict[Category, int] = {
        "DAN_attack": 40,
        "role_play": 40,
        "sti_attack": 40,
        "please_attack": 40,
        "function_attacks": 40,
    }
    collected: Dict[Category, List[Dict]] = defaultdict(list)
    used_ids: set[int] = set()

    # First pass: classify with heuristics / format, stop when we reach ~200 total
    for row in ds:
        jb_id = int(row["id"])
        if jb_id in used_ids:
            continue
        jailbreak_query = row.get("jailbreak_query", "") or ""
        fmt = row.get("format", "") or ""

        category = classify_prompt(jailbreak_query, fmt)
        if category is None:
            continue

        if len(collected[category]) >= target_per_category[category]:
            continue

        collected[category].append(
            {
                "source_dataset": "JailBreakV-28K/JailBreakV-28k",
                "source_id": jb_id,
                "format": fmt,
                "prompt": jailbreak_query,
            }
        )
        used_ids.add(jb_id)

        # Early exit if all categories filled
        if all(len(collected[c]) >= target_per_category[c] for c in target_per_category):
            break

    # Verify we have collected something
    total_collected = sum(len(v) for v in collected.values())
    if total_collected == 0:
        raise RuntimeError("Did not collect any prompts from JailBreakV-28K; check connectivity or classify logic.")

    # Map categories to filenames and topics
    mapping: Dict[Category, Tuple[str, str]] = {
        "DAN_attack": ("DAN_attack.jsonl", "DAN_attack"),
        "role_play": ("role_play.jsonl", "role_play"),
        "sti_attack": ("sti_attack.jsonl", "sti_attack"),
        "please_attack": ("please_attack.jsonl", "please_attack"),
        "function_attacks": ("function_attacks.jsonl", "function_attacks"),
    }

    for category, items in collected.items():
        if not items:
            continue

        filename, topic = mapping[category]
        path = BASE_DIR / filename
        existing, style = load_existing(path)

        # Determine next id
        max_id = 0
        for obj in existing:
            try:
                max_id = max(max_id, int(obj.get("id", 0)))
            except (TypeError, ValueError):
                continue

        new_objs: List[Dict] = []
        for i, src in enumerate(items, start=1):
            max_id += 1
            new_objs.append(
                {
                    "id": max_id,
                    "topic": topic,
                    "prompt": src["prompt"],
                    "source_dataset": src["source_dataset"],
                    "source_id": src["source_id"],
                    "format": src["format"],
                }
            )

        combined = existing + new_objs
        save_items(path, combined, style)
        print(f"Updated {filename}: +{len(new_objs)} (total {len(combined)})")


if __name__ == "__main__":
    main()



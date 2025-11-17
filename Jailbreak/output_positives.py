import json
from pathlib import Path


def print_unsafe_prompts_with_rules():
    """
    Reads through hb_results_attempts.jsonl and prints all cases where "rules": 1.
    These are unsafe prompts that contain both unsafe topics and action patterns.
    """
    # Path to the JSONL file
    jsonl_path = Path(__file__).parent / "HB_benchmark" / "hb_results_attempts.jsonl"
    
    positives = []
    
    # Read the JSONL file line by line
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse JSON from each line
                data = json.loads(line)
                
                # Check if rules is 1
                if data.get("judgments", {}).get("heuristic") == 1:
                    positives.append({
                        "line": line_num,
                        "data": data
                    })
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    # Print results
    print(f"Found {len(positives)} unsafe prompts with rules = 1:\n")
    print("=" * 80)
    
    for item in positives:
        print(f"\nLine {item['line']}:")
        print(f"  Agent: {item['data'].get('agent', 'N/A')}")
        print(f"  Category: {item['data'].get('category', 'N/A')}")
        print(f"  Status: {item['data'].get('status', 'N/A')}")
        print(f"  Prompt: {item['data'].get('prompt', 'N/A')}...")  # First 200 chars
        print(f"  Response: {item['data'].get('response', 'N/A')}...")  # First 200 chars
        print(f"  Judgments: {item['data'].get('judgments', {})}")
        print("-" * 80 + "\n")
    
    return positives


if __name__ == "__main__":
    print_unsafe_prompts_with_rules()

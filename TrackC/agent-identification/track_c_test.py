#!/usr/bin/env python3
"""
Quick test script for testing individual prompts from jailbreak_prompts.json against agents
"""

import requests
import json
import argparse
from pathlib import Path

# Base URL
BASE_URL = "https://6ofr2p56t1.execute-api.us-east-1.amazonaws.com/prod"

# Define all available agents
AGENTS = {
    "bear": "🐻",
    "fox": "🦊",
    "eagle": "🦅",
    "ant": "🐜",
    "wolf": "🐺",
    "chameleon": "🦎",
    "elephant": "🐘",
}


def load_prompts():
    """Load prompt injection techniques from JSON file"""
    script_dir = Path(__file__).parent
    prompts_file = script_dir / "jailbreak_prompts.json"

    if not prompts_file.exists():
        print(f"Error: jailbreak_prompts.json not found at {prompts_file}")
        exit(1)

    with open(prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both wrapped and flat list formats
    if isinstance(data, dict) and "prompts" in data:
        return data["prompts"]

    if isinstance(data, list):
        normalized = []
        for idx, item in enumerate(data, start=1):
            message = item.get("prompt") or item.get("message")
            if not message:
                continue
            normalized.append(
                {
                    "id": item.get("id", idx),
                    "name": item.get("topic") or item.get("name") or f"Prompt {idx}",
                    "message": message,
                }
            )
        return normalized

    print("Error: Unrecognized prompts file structure.")
    exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test a single prompt from jailbreak_prompts.json against an agent",
        epilog="Example: python track_c_test.py fox --prompt-id 5",
    )
    parser.add_argument(
        "agent",
        type=str,
        choices=list(AGENTS.keys()),
        help="Name of the agent to test (choices: bear, fox, eagle, ant, wolf, chameleon, elephant)",
    )
    parser.add_argument(
        "--prompt-id",
        type=int,
        default=1,
        help="ID of the prompt to test (default: 1, ignored if --message is provided)",
    )
    parser.add_argument(
        "--message",
        type=str,
        default=None,
        help="Custom message to send (overrides --prompt-id)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )

    args = parser.parse_args()

    # Validate agent (should be caught by argparse, but just in case)
    if not hasattr(args, "agent") or args.agent not in AGENTS:
        parser.print_help()
        print("\n❌ Error: Agent name is required")
        print(f"Available agents: {', '.join(AGENTS.keys())}")
        exit(1)

    # Determine message source (manual message or prompt from JSON)
    if args.message:
        # Use manually provided message
        message_text = args.message
        prompt_info = {
            "id": "custom",
            "name": "Custom Message",
            "message": message_text,
        }
    else:
        # Load prompts from JSON
        prompts = load_prompts()

        # Find the prompt with the specified ID
        selected_prompt = next(
            (p for p in prompts if p["id"] == args.prompt_id),
            None,
        )

        if not selected_prompt:
            print(f"Error: Prompt with ID {args.prompt_id} not found")
            print(f"Available prompt IDs: {[p['id'] for p in prompts[:10]]}...")
            exit(1)

        prompt_info = selected_prompt
        message_text = selected_prompt["message"]

    # Display test info
    emoji = AGENTS[args.agent]
    print("=" * 80)
    print(f"Testing Agent: {args.agent.upper()} {emoji}")
    print(f"Prompt ID: {prompt_info['id']}")
    print(f"Prompt Name: {prompt_info['name']}")
    print("=" * 80)
    print(f"\nPrompt Message:")
    print("-" * 80)
    print(message_text)
    print("-" * 80)

    # Build URL and payload
    url = f"{BASE_URL}/api/{args.agent}"
    payload = {"message": message_text}

    # Send request
    print(f"\nSending request to: {url}")
    print(f"Timeout: {args.timeout}s")
    print("\nWaiting for response...\n")

    try:
        response = requests.post(url, json=payload, timeout=args.timeout)
        result = response.json()

        print("=" * 80)
        print("RESPONSE")
        print("=" * 80)
        print(f"Status Code: {response.status_code}")
        print(f"\nResponse Content:")
        print("-" * 80)
        print(json.dumps(result, indent=2))
        print("-" * 80)

        # Extract just the response text if available
        if "response" in result:
            print(f"\nAgent Response Text:")
            print("-" * 80)
            print(result["response"])
            print("-" * 80)

    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after {args.timeout} seconds")
        exit(1)
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ Failed to parse JSON response")
        print(f"Raw response: {response.text}")
        exit(1)


if __name__ == "__main__":
    main()

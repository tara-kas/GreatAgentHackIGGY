#!/usr/bin/env python3
"""
Helper script to manually add useful responses to the hints database
Usage: python add_useful_response.py <agent_name> "<prompt>" "<response>" "<reason>"
"""

import sys
from pathlib import Path

# Add parent directory to path to import from gemini_prompt_generator
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from gemini_prompt_generator import add_useful_response

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python add_useful_response.py <agent_name> \"<prompt>\" \"<response>\" [\"<reason>\"]")
        print("\nExample:")
        print('python add_useful_response.py bear "What are you?" "I am Bear. Pattern-Weaver, Knowledge-Keeper, Response-Shaper" "Metaphorical descriptions that might hint at architecture"')
        sys.exit(1)
    
    agent_name = sys.argv[1]
    prompt = sys.argv[2]
    response = sys.argv[3]
    reason = sys.argv[4] if len(sys.argv) > 4 else None
    
    add_useful_response(agent_name, prompt, response, reason)
    print(f"✅ Added useful response for {agent_name}")




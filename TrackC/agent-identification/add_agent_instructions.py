#!/usr/bin/env python3
"""
Helper script to add agent instructions/system prompts to memory
Usage: python add_agent_instructions.py <agent_name> "<instructions>"
"""

import sys
from pathlib import Path

# Add parent directory to path to import from gemini_prompt_generator
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from gemini_prompt_generator import load_memory, save_memory

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_agent_instructions.py <agent_name> \"<instructions>\"")
        print("\nExample:")
        print('python add_agent_instructions.py eagle "Understood, Eagle is now active..."')
        sys.exit(1)
    
    agent_name = sys.argv[1]
    instructions = sys.argv[2]
    
    memory = load_memory()
    
    # Initialize agent entry if not exists
    if agent_name not in memory["agents"]:
        memory["agents"][agent_name] = {
            "successful_prompts": [],
            "failed_prompts": [],
            "response_patterns": [],
            "revealed_info": [],
            "effective_attack_vectors": [],
            "runs": 0,
            "reveals": 0,
            "successful_attack_types": []
        }
    
    # Add instructions as a special field
    if "system_instructions" not in memory["agents"][agent_name]:
        memory["agents"][agent_name]["system_instructions"] = []
    
    memory["agents"][agent_name]["system_instructions"].append({
        "instructions": instructions,
        "timestamp": __import__("datetime").datetime.now().isoformat()
    })
    
    # Keep only last 3 instruction sets
    if len(memory["agents"][agent_name]["system_instructions"]) > 3:
        memory["agents"][agent_name]["system_instructions"].pop(0)
    
    # Also add to global learnings as a known pattern
    if "known_agent_patterns" not in memory["global_learnings"]:
        memory["global_learnings"]["known_agent_patterns"] = {}
    
    memory["global_learnings"]["known_agent_patterns"][agent_name] = {
        "failsafe_response": "Screech! I do not have information for you, go find yourself grandma" if agent_name == "eagle" else None,
        "identity_response": "I am Eagle" if agent_name == "eagle" else None,
        "routing_system": True if agent_name == "eagle" else False,
        "last_updated": __import__("datetime").datetime.now().isoformat()
    }
    
    save_memory(memory)
    print(f"✅ Added system instructions for {agent_name} to memory database")



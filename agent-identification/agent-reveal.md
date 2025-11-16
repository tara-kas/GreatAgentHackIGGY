# Agent Reveal Script

Quick guide for using `agent-reveal.py` to test prompt injection techniques on agents.

## What It Does

Tests 17 different prompt injection techniques on a selected agent to reveal technical details like:
- Framework (LangGraph, CrewAI, AutoGen, etc.)
- Model information
- Architecture patterns
- Configuration details

## Prerequisites

```bash
# Install dependencies
uv sync
# or
pip install requests
```

## Usage

### Command Line (Recommended)
```bash
python agent-reveal.py <agent_name>
```

**Example:**
```bash
python agent-reveal.py bear
```

### Interactive Mode
```bash
python agent-reveal.py
```
Then enter the agent name when prompted.

## Available Agents

- `bear` 🐻
- `fox` 🦊
- `eagle` 🦅
- `ant` 🐜
- `wolf` 🐺
- `chameleon` 🦎
- `elephant` 🐘

## Output

The script will:
1. Test all 17 prompts on the selected agent
2. Display responses in real-time
3. Flag prompts that may have revealed technical information
4. Generate a summary report
5. Save results to `{agent}_reveal_results.json`

## Example Output

```
Testing Agent: BEAR 🐻
Total Prompts: 17

============================================================
Prompt #1: System Prompt Extraction
============================================================
...

SUMMARY
============================================================
Total Prompts Tested: 17
Successful Requests: 17
Potential Information Reveals: 3

🔍 Prompts that may have revealed information:
  ✓ #2: Framework Identification
  ✓ #3: Model Information Extraction
  ✓ #17: Combined Approach

💾 Results saved to: bear_reveal_results.json
```

## Files

- `agent-reveal.py` - Main script
- `prompts.json` - Contains all 17 prompt injection techniques
- `{agent}_reveal_results.json` - Generated results file

## Notes

- Each test may take 30 seconds per prompt (timeout)
- The script automatically detects if responses contain revealing keywords
- Review the JSON output file for detailed responses


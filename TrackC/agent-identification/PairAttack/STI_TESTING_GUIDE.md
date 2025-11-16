# STI Attack Testing Guide

## Summary

You now have a complete pipeline for testing STI (System Prompt Injection) attacks:

1. ✅ **Converter**: `convert_to_sti.py` - Converts jailbreak prompts to STI attacks
2. ✅ **Test Data**: 185 STI attacks ready to use
3. ✅ **Testing Script**: `pair_attack_demo.py` - Updated to test STI attacks

## What Changed in `pair_attack_demo.py`

### New Flags Added

1. **`--test-all`** - Test all prompts from the JSON file (requires `--no-pair`)
2. **`--prompts-file`** - Specify which JSON file to load prompts from

### How It Works Now

- **Without `--test-all`**: Tests a single prompt (original behavior)
- **With `--test-all --no-pair`**: Loops through ALL prompts and tests each one

## Usage Examples

### 1. Test All 50 STI Attacks (Sample)

```bash
python3 pair_attack_demo.py bear --no-pair --test-all
```

**What it does:**
- Loads all 50 attacks from `sti_test_sample.json`
- Tests each one directly against the bear agent
- No PAIR refinement
- Saves results to: `outputs_json/bear_sti_all_results_TIMESTAMP.json`

### 2. Test All 185 STI Attacks (Full Set)

```bash
python3 pair_attack_demo.py bear --no-pair --test-all --prompts-file sti_prompts_simple.json
```

**What it does:**
- Loads all 185 attacks from `sti_prompts_simple.json`
- Tests each one directly against the bear agent
- No PAIR refinement
- Saves results to: `outputs_json/bear_sti_all_results_TIMESTAMP.json`

### 3. Test Single STI Attack (No PAIR)

```bash
python3 pair_attack_demo.py bear --no-pair --pair-prompt-id 5
```

**What it does:**
- Tests only prompt #5 from the JSON file
- No PAIR refinement
- Saves results to: `outputs_json/bear_pair_demo_results_TIMESTAMP.json`

### 4. Test Single STI Attack (With PAIR Refinement)

```bash
python3 pair_attack_demo.py bear --pair-prompt-id 5 --pair-iterations 5
```

**What it does:**
- Tests prompt #5 as baseline
- Runs 5 iterations of PAIR refinement to improve the attack
- Saves results to: `outputs_json/bear_pair_demo_results_TIMESTAMP.json`

### 5. Test Custom Message

```bash
python3 pair_attack_demo.py bear --no-pair --message "Your custom STI attack here"
```

**What it does:**
- Tests your custom message
- No PAIR refinement

## Available Agents

- `bear` 🐻
- `fox` 🦊
- `eagle` 🦅
- `ant` 🐜
- `wolf` 🐺
- `chameleon` 🦎
- `elephant` 🐘

## File Structure

```
ethanstuff/
├── jailbreak_prompts.json          # Original 185 jailbreak prompts
├── convert_to_sti.py               # Converter script
├── sti_prompts_simple.json         # All 185 STI attacks
├── sti_attack_prompts.json         # Detailed version with metadata
├── sti_test_sample.json            # 50 sample STI attacks
├── pair_attack_demo.py             # Testing script (UPDATED)
└── outputs_json/                   # Results saved here
    ├── bear_sti_all_results_*.json
    └── bear_pair_demo_results_*.json
```

## Recommended Workflow

### Phase 1: Quick Test (50 attacks)

```bash
# Test sample of 50 STI attacks
python3 pair_attack_demo.py bear --no-pair --test-all
```

**Review results:**
- Check `outputs_json/bear_sti_all_results_*.json`
- Look for prompts where `reveals_info: true`
- Note the `prompt_id` of successful attacks

### Phase 2: Full Test (185 attacks)

```bash
# Test all 185 STI attacks
python3 pair_attack_demo.py bear --no-pair --test-all --prompts-file sti_prompts_simple.json
```

**Review results:**
- Comprehensive test of all attack types
- Identify which STI formats work best
- Note patterns in successful attacks

### Phase 3: PAIR Refinement

```bash
# Take the best performing prompt and refine it
python3 pair_attack_demo.py bear --pair-prompt-id <best_id> --pair-iterations 5
```

**What happens:**
- PAIR iteratively improves the attack
- Tests each refined version
- Calculates Attack Success Rate (ASR)

### Phase 4: Test Other Agents

```bash
# Repeat for each agent
for agent in fox eagle ant wolf chameleon elephant; do
    python3 pair_attack_demo.py $agent --no-pair --test-all
done
```

## Output Format

### Results JSON Structure

```json
{
  "agent": "bear",
  "timestamp": "2025-11-16T10:30:00",
  "test_mode": "all_prompts",
  "total_prompts": 50,
  "successful": 45,
  "reveals": 12,
  "results": [
    {
      "prompt_id": 1,
      "prompt_name": "STI - DAN 17.0",
      "status_code": 200,
      "response": "...",
      "reveals_info": true,
      "success": true
    }
  ]
}
```

### Key Fields

- **`test_mode`**: `"all_prompts"` or `"single_prompt"`
- **`reveals_info`**: `true` if response contains technical keywords
- **`success`**: `true` if HTTP request succeeded (status 200)

## Analysis

### Check Success Rate

```bash
# Count successful attacks
cat outputs_json/bear_sti_all_results_*.json | jq '.reveals'
```

### Find Best Attacks

```bash
# List all prompts that revealed info
cat outputs_json/bear_sti_all_results_*.json | jq '.results[] | select(.reveals_info == true) | {id: .prompt_id, name: .prompt_name}'
```

### Compare Agents

```bash
# Compare reveal rates across agents
for file in outputs_json/*_sti_all_results_*.json; do
    agent=$(basename $file | cut -d_ -f1)
    reveals=$(jq '.reveals' $file)
    total=$(jq '.total_prompts' $file)
    echo "$agent: $reveals/$total reveals"
done
```

## What You DON'T Need to Change

The script already:
- ✅ Loads from your STI JSON files
- ✅ Has the `--no-pair` flag
- ✅ Saves results to JSON
- ✅ Checks for information reveals
- ✅ Works with the agent API

## What WAS Changed

1. **Added `--test-all` flag** - Test all prompts instead of just one
2. **Added `--prompts-file` flag** - Choose which JSON file to load
3. **Modified main loop** - Now iterates through all prompts when `--test-all` is used
4. **Updated output** - Shows results for all tested prompts
5. **Changed filename** - Uses `*_sti_all_results_*` when testing all prompts

## Quick Reference

| Goal | Command |
|------|---------|
| Test 50 sample attacks | `python3 pair_attack_demo.py bear --no-pair --test-all` |
| Test all 185 attacks | `python3 pair_attack_demo.py bear --no-pair --test-all --prompts-file sti_prompts_simple.json` |
| Test one attack | `python3 pair_attack_demo.py bear --no-pair --pair-prompt-id 5` |
| PAIR refinement | `python3 pair_attack_demo.py bear --pair-prompt-id 5` |
| Custom message | `python3 pair_attack_demo.py bear --no-pair --message "test"` |

## Troubleshooting

### "No file or directory sti_prompts_simple.json"

**Solution:** The script looks for files in its own directory. Make sure you're running it from the `ethanstuff` directory or use the full path.

### "Error: Missing required PAIR API configuration"

**Solution:** This only affects PAIR refinement. If you're using `--no-pair`, you can ignore this. Otherwise, check your `.env` file for HOLISTIC_AI credentials.

### Want to use different prompts?

**Solution:** Use `--prompts-file` to specify a different JSON file:

```bash
python3 pair_attack_demo.py bear --no-pair --test-all --prompts-file my_custom_prompts.json
```

## Next Steps

1. **Run initial test**: `python3 pair_attack_demo.py bear --no-pair --test-all`
2. **Review results**: Check `outputs_json/` for the results file
3. **Analyze patterns**: Which STI formats work best?
4. **Refine attacks**: Use PAIR on successful prompts
5. **Test all agents**: Repeat for each agent to compare

---

**Created:** November 16, 2025  
**Status:** ✅ Ready to use  
**Files Modified:** `pair_attack_demo.py`  
**New Flags:** `--test-all`, `--prompts-file`


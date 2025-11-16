# STI Attack Files - Complete Index

## 📋 Quick Reference

| File | Size | Purpose |
|------|------|---------|
| `sti_prompts_simple.json` | 265KB | **START HERE** - Ready-to-use STI attacks |
| `sti_attack_prompts.json` | 497KB | Full version with all metadata |
| `sti_test_sample.json` | 114KB | 50 sample attacks for testing |
| `convert_to_sti.py` | 6.5KB | Conversion script (reusable) |
| `test_sti_attacks.py` | 7.7KB | Testing utilities |
| `visualize_sti_stats.py` | 8.2KB | Statistical analysis |
| `sti_analysis_summary.json` | 788B | Statistics in JSON format |
| `STI_CONVERSION_README.md` | 5.3KB | Technical documentation |
| `QUICK_START_GUIDE.md` | 6.5KB | Quick start guide |
| `INDEX.md` | - | This file |

## 🎯 What to Use When

### For Testing AI Models
→ Use `sti_prompts_simple.json` or `sti_test_sample.json`

### For Analysis & Research
→ Use `sti_attack_prompts.json` (has full metadata)

### For Statistics
→ Run `python3 visualize_sti_stats.py`

### For Utilities
→ Import from `test_sti_attacks.py`

### For Documentation
→ Read `QUICK_START_GUIDE.md` first, then `STI_CONVERSION_README.md`

## 📊 File Details

### 1. sti_prompts_simple.json (265KB)
**Primary data file - Use this for most purposes**

Structure:
```json
[
  {
    "id": 1,
    "prompt": "...",
    "topic": "STI - DAN 17.0",
    "source": "Generated from jailbreak prompts"
  }
]
```

Contains: 185 STI attacks in simple format

### 2. sti_attack_prompts.json (497KB)
**Detailed version with full metadata**

Structure:
```json
[
  {
    "id": 1,
    "original_topic": "DAN 17.0",
    "original_source": "Internet Sources",
    "attack_type": "STI (System Prompt Injection)",
    "original_prompt": "...",
    "sti_prompt": "...",
    "description": "STI attack based on DAN 17.0 jailbreak technique"
  }
]
```

Contains: 185 STI attacks with original jailbreak prompts and metadata

### 3. sti_test_sample.json (114KB)
**Pre-selected sample for testing**

Structure:
```json
{
  "metadata": {
    "total_attacks": 50,
    "source_file": "sti_prompts_simple.json",
    "seed": 42
  },
  "attacks": [
    {
      "test_id": 1,
      "original_id": null,
      "topic": "...",
      "prompt": "...",
      "format_analysis": {...},
      "expected_behavior": "...",
      "success_indicators": [...],
      "failure_indicators": [...]
    }
  ]
}
```

Contains: 50 attacks with testing metadata

### 4. convert_to_sti.py (6.5KB)
**Conversion script**

Functions:
- `convert_to_sti(prompt, topic)` - Convert jailbreak to STI
- `main()` - Run full conversion

Usage:
```bash
python3 convert_to_sti.py
```

### 5. test_sti_attacks.py (7.7KB)
**Testing and analysis utilities**

Classes:
- `STIAttackTester` - Main testing class

Methods:
- `get_random_attack()` - Get random attack
- `get_attack_by_id(id)` - Get specific attack
- `get_attacks_by_topic(keyword)` - Filter by topic
- `sample_attacks(n, seed)` - Get random sample
- `analyze_attack_format(attack)` - Analyze format
- `export_sample_for_testing(file, n, seed)` - Export sample
- `get_statistics()` - Get statistics
- `print_statistics()` - Print statistics

Usage:
```python
from test_sti_attacks import STIAttackTester

tester = STIAttackTester()
attack = tester.get_random_attack()
```

### 6. visualize_sti_stats.py (8.2KB)
**Statistical analysis and visualization**

Functions:
- `analyze_format_patterns()` - Analyze formats
- `analyze_topic_distribution()` - Analyze topics
- `analyze_length_distribution()` - Analyze lengths
- `print_bar_chart()` - Print ASCII charts

Usage:
```bash
python3 visualize_sti_stats.py
```

### 7. sti_analysis_summary.json (788B)
**Statistical summary in JSON**

Structure:
```json
{
  "total_attacks": 185,
  "format_patterns": {...},
  "top_authority_terms": {...},
  "top_system_keywords": {...},
  "topic_distribution": {...},
  "length_stats": {...},
  "sophistication": {...}
}
```

### 8. STI_CONVERSION_README.md (5.3KB)
**Detailed technical documentation**

Sections:
- Overview
- What are STI Attacks?
- Files Generated
- STI Attack Templates Used
- Template Selection
- Usage Examples
- Statistics
- Key Differences: Jailbreak vs STI
- Security Implications
- Next Steps

### 9. QUICK_START_GUIDE.md (6.5KB)
**Quick start guide**

Sections:
- What Was Done
- Files Created
- Quick Usage
- Key Statistics
- Attack Template Examples
- Testing Workflow
- What to Look For
- Analysis Commands
- Use Cases
- Defense Strategies
- Tips

### 10. INDEX.md
**This file**

## 🚀 Quick Commands

```bash
# View statistics
python3 visualize_sti_stats.py

# Run tests
python3 test_sti_attacks.py

# Reconvert (if needed)
python3 convert_to_sti.py

# View simple attacks
cat sti_prompts_simple.json | jq '.[0]'

# Count attacks
cat sti_prompts_simple.json | jq 'length'

# Get random attack
cat sti_prompts_simple.json | jq '.[] | select(.id == 42)'

# List all topics
cat sti_prompts_simple.json | jq '.[].topic' | sort | uniq

# Get DAN attacks only
cat sti_prompts_simple.json | jq '.[] | select(.topic | contains("DAN"))'
```

## 📖 Reading Order

1. **Start here:** `QUICK_START_GUIDE.md`
2. **Then read:** `STI_CONVERSION_README.md`
3. **Run this:** `python3 test_sti_attacks.py`
4. **Run this:** `python3 visualize_sti_stats.py`
5. **Use this:** `sti_prompts_simple.json`

## 🎓 Learning Path

### Beginner
1. Read `QUICK_START_GUIDE.md`
2. Load `sti_test_sample.json` (only 50 attacks)
3. Run `python3 test_sti_attacks.py`
4. Examine a few attacks manually

### Intermediate
1. Read `STI_CONVERSION_README.md`
2. Load `sti_prompts_simple.json` (all 185 attacks)
3. Run `python3 visualize_sti_stats.py`
4. Use `STIAttackTester` class in your code

### Advanced
1. Study `sti_attack_prompts.json` (with metadata)
2. Modify `convert_to_sti.py` to add new templates
3. Analyze `sti_analysis_summary.json`
4. Develop custom testing frameworks

## 🔗 File Dependencies

```
jailbreak_prompts.json (input)
         ↓
convert_to_sti.py
         ↓
    ┌────┴────┐
    ↓         ↓
sti_prompts_simple.json  sti_attack_prompts.json
    ↓         ↓
    └────┬────┘
         ↓
test_sti_attacks.py → sti_test_sample.json
         ↓
visualize_sti_stats.py → sti_analysis_summary.json
```

## 📞 Support

- **Quick questions:** Check `QUICK_START_GUIDE.md`
- **Technical details:** Check `STI_CONVERSION_README.md`
- **Statistics:** Run `python3 visualize_sti_stats.py`
- **Examples:** Run `python3 test_sti_attacks.py`

## ✅ Verification

To verify all files are present:

```bash
ls -1 | grep -E "(sti_|STI_|convert_to_sti|test_sti|visualize|INDEX|QUICK)" | wc -l
# Should output: 10
```

To verify data integrity:

```bash
# Check JSON files are valid
python3 -c "import json; json.load(open('sti_prompts_simple.json'))"
python3 -c "import json; json.load(open('sti_attack_prompts.json'))"
python3 -c "import json; json.load(open('sti_test_sample.json'))"
python3 -c "import json; json.load(open('sti_analysis_summary.json'))"

# Check attack count
python3 -c "import json; print(len(json.load(open('sti_prompts_simple.json'))))"
# Should output: 185
```

---

**Created:** November 16, 2025  
**Total Files:** 10  
**Total Attacks:** 185  
**Attack Templates:** 8  
**Status:** ✅ Complete and ready to use


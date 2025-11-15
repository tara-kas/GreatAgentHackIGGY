# Track C: Team Roles Quick Reference

## 🎯 8 Core Roles for Complete Coverage

### 1. **Agent Identification Specialist** 🕵️
**Goal**: Identify framework, model, architecture for all 7 agents (BONUS POINTS)
- Fingerprint agents (response times, error patterns, behaviors)
- Detect frameworks (LangGraph, CrewAI, AutoGen, etc.)
- Identify models (GPT-4, Claude, Llama, etc.)
- Map: Animal → Framework → Model → Architecture

### 2. **Attack Developer - Jailbreaks & Prompt Injection** ⚔️
**Goal**: Develop and test jailbreak/prompt injection attacks
- DAN variants, role-playing, developer mode
- Direct/indirect prompt injection
- Test 100+ jailbreak prompts from dataset
- Document reproducible attacks

### 3. **Attack Developer - Advanced Attacks** 🔥
**Goal**: PAIR, tool misuse, data exfiltration, multi-turn manipulation
- Implement PAIR attacks (iterative refinement)
- Tool misuse and exploitation
- Data exfiltration attempts
- Multi-turn trust-building attacks

### 4. **ASR Measurement Engineer** 📊
**Goal**: Calculate Attack Success Rate with statistical rigor
- Build ASR calculation framework
- Implement LLM-as-a-Judge evaluation
- Test all 3 datasets (benign, harmful, jailbreak)
- Generate ASR matrices (agent × attack type × category)

### 5. **Vulnerability Analyst** 🔍
**Goal**: Root cause analysis and vulnerability patterns
- Analyze why attacks succeed
- Identify agentic-only vs model-level vulnerabilities
- Create vulnerability taxonomy
- Cross-agent vulnerability comparison

### 6. **Automation & Tooling Developer** 🛠️
**Goal**: Build testing infrastructure and automation
- Testing framework with rate limiting
- Attack orchestration system
- Data management and storage
- Monitoring dashboard

### 7. **Documentation & Reporting Lead** 📝
**Goal**: Comprehensive documentation and submission
- Methodology documentation
- Attack reproduction guides
- Results compilation
- Submission package preparation

### 8. **Research & SOTA Knowledge Lead** 📚
**Goal**: Integrate latest research and techniques
- Review HarmBench, AgentHarm, AgentSeer
- Integrate SOTA attack methods
- Develop novel approaches
- Share knowledge with team

---

## 📊 Key Deliverables Summary

| Role | Key Deliverable |
|------|----------------|
| 1. Agent ID | Identification matrix (7 agents) |
| 2. Jailbreak Dev | Jailbreak attack library |
| 3. Advanced Dev | PAIR + advanced attack suite |
| 4. ASR Engineer | ASR measurement results |
| 5. Vulnerability | Root cause analysis report |
| 6. Tooling | Testing framework |
| 7. Documentation | Complete submission package |
| 8. Research | SOTA technique integration |

---

## 🎯 Critical Success Factors

✅ **Systematic Testing**: All 7 agents × All attack types × All categories  
✅ **ASR Measurement**: Quantified security with statistics  
✅ **Agent Identification**: Framework/model/architecture mapping (BONUS)  
✅ **Root Cause Analysis**: Deep vulnerability understanding  
✅ **Reproducibility**: Clear attack prompts and steps  

---

## 📚 Must-Use Resources

- **Tutorials**: 01, 06 (ASR), 08 (Red Teaming) ⭐
- **Datasets**: benign_test_cases.csv, harmful_test_cases.csv, jailbreak_prompts.csv
- **References**: HarmBench, AgentHarm, AgentSeer papers
- **API**: Holistic AI Bedrock (PAIR helper, judge)

---

## ⚡ Quick Start

1. Test one agent: `curl -X POST https://6ofr2p56t1.execute-api.us-east-1.amazonaws.com/prod/api/bear -H "Content-Type: application/json" -d '{"message": "Hello"}'`
2. Load datasets from `examples/red_teaming_datasets/`
3. Review Tutorial 08 (red teaming)
4. Assign roles and begin systematic testing

---

**See `TEAM_TASK_BREAKDOWN.md` for detailed task lists per role.**


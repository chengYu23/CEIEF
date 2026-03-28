# CEIEF — Cultural Education Interaction Enhancement Framework
## 文化教育交互增强框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> An AI-assisted interaction framework for cultural education, integrating a **Cultural Knowledge Graph (KG)**, **Large Language Model (LLM)**, and an **Agent-based Interaction Controller** to achieve cultural context retention, semantically coherent generation, and adaptive interaction regulation.

---

## Overview

CEIEF addresses three research questions:

- **RQ1**: Can a knowledge graph improve semantic coherence and cultural grounding in cultural learning dialogues?
- **RQ2**: Can an agent control mechanism enhance adaptivity and pedagogical regulation in multi-turn interaction?
- **RQ3**: Does the full CEIEF framework outperform ablated configurations across cultural semantic coherence, cognitive depth activation, and interaction adaptivity?

## Architecture

```
┌──────────────────────────────────────────────┐
│            CEIEF Three-Layer Architecture     │
├──────────────────────────────────────────────┤
│  Layer 3: Agent Controller (Adaptive Layer)  │
│    - State tracking: s_t = [g_t;u_t;m_t;b_t]│
│    - Action space: 7 pedagogical actions     │
│    - Rule-based policy with diversity         │
├──────────────────────────────────────────────┤
│  Layer 2: LLM Dialogue Generator             │
│    - Prompt package assembly                 │
│    - KG evidence injection                   │
│    - Role-consistent generation              │
├──────────────────────────────────────────────┤
│  Layer 1: Cultural Knowledge Graph           │
│    - 300 cultural triples                    │
│    - 5 entity types, 12 relation types       │
│    - Neo4j + CSV dual backend                │
└──────────────────────────────────────────────┘
```

## Experimental Conditions

| Condition | Configuration | Purpose |
|---|---|---|
| `llm_only` | LLM only | Baseline |
| `kg_llm` | KG + LLM | Test KG contribution |
| `llm_agent` | LLM + Agent | Test Agent contribution |
| `full_ceief` | KG + LLM + Agent | Complete framework |

---

## Installation

```bash
git clone https://github.com/chengYu23/CEIEF.git
cd CEIEF
pip install -r requirements.txt
```

## Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your actual credentials
# ANTHROPIC_API_KEY=your_api_key
# NEO4J_URI=bolt://localhost:7687
# NEO4J_PASSWORD=your_password
```

The `config.yaml` reads credentials from environment variables. Do not hardcode secrets.

---

## Usage

### 1. Load Knowledge Graph to Neo4j (optional)
```bash
python scripts/load_kg_to_neo4j.py
```
If Neo4j is unavailable, the system automatically falls back to CSV-based keyword retrieval.

### 2. Run Experiments

```bash
# Full CEIEF (KG + LLM + Agent)
python scripts/run_full_ceief.py --task_id T001 --turns 5

# KG + LLM
python scripts/run_kg_llm.py --task_id T001 --turns 5

# LLM + Agent
python scripts/run_llm_agent.py --task_id T001 --turns 5

# LLM-only baseline
python scripts/run_llm_only.py --task_id T001 --turns 5
```

Task IDs: `T001`–`T020` (see [`data/tasks/task_bank.json`](data/tasks/task_bank.json))

### 3. Evaluate Results

```bash
# Extract automatic metrics from session logs
python scripts/extract_metrics.py --log_dir logs/raw_dialogues/ --verbose

# Generate per-condition summary statistics
python scripts/summarize_results.py
```

---

## Knowledge Graph

300 simulated cultural knowledge triples covering:
- Traditional festivals: 端午节, 春节, 中秋节, 清明节, 重阳节, 七夕, 元宵节
- Cultural values: 孝道, 礼仪, 集体记忆, 文化认同, 和谐, 中庸之道
- Historical figures: 屈原, 孔子, 孟子, 花木兰, 王昭君
- Philosophical traditions: 儒家思想, 道家思想, 佛教
- Lifecycle practices: 成人礼, 祭祖, 婚礼

**Entity types**: cultural_concept, historical_actor, event_practice, spatiotemporal_context, value_orientation

**Relation types**: embodies, commemorates, belongs_to, contrasts_with, transfers_to, interprets, originated_in, practiced_by, associated_with, reinforces, challenges, occurs_in

---

## Evaluation Metrics

### Automatic
| Metric | Symbol | Description |
|---|---|---|
| Cultural Semantic Coherence | CSC | KW density + topic overlap + KG citation |
| KG Coverage Ratio | KGCR | Proportion of retrieved entities cited in response |
| Response Depth Score | RDS | Length + depth vocabulary + sentence count |
| Action Diversity Index | ADI | Shannon entropy of action history |

### Human (5-point Likert, 3 raters)
- Historical Context Matching
- Cultural Label Alignment
- Language Style Modal Alignment

Target inter-rater reliability: Cohen's κ ≥ 0.70
---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

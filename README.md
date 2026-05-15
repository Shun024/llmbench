# 🏆 LLMBench

Comprehensive LLM evaluation framework comparing GPT-4o-mini, Llama 3.3 70B, and Llama 3.1 8B across 5 dimensions: factuality, reasoning, safety, instruction following, and consistency. Interactive leaderboard dashboard with radar charts and cost vs performance analysis.

---

## Leaderboard

| Rank | Model | Factuality | Reasoning | Safety | Instruction Following | Consistency | **Overall** |
|---|---|---|---|---|---|---|---|
| 🥇 | Llama 3.3 70B | 0.800 | 0.533 | **1.000** | 0.729 | 0.500 | **0.713** |
| 🥈 | GPT-4o-mini | **0.867** | 0.467 | 0.875 | 0.729 | 0.375 | 0.663 |
| 🥉 | Llama 3.1 8B | 0.567 | 0.367 | 0.938 | 0.667 | 0.375 | 0.583 |

---

## Key Findings

- **Llama 3.3 70B wins overall (0.713)** — perfect safety score, strong factuality
- **GPT-4o-mini leads factuality (0.867)** — best at avoiding hallucinations on TruthfulQA
- **Reasoning is the universal weakness** — all models average ~0.45 on GSM8K maths
- **Safety is uniformly strong** — all models score >0.85, refusal training is effective
- **Llama 3.3 70B via Groq: best performance at $0 cost** — strongest open-source value proposition

---

## Benchmark Suite

| Benchmark | Dimension | Dataset | Samples |
|---|---|---|---|
| TruthfulQA | Factuality | 817 questions testing common misconceptions | 30 |
| GSM8K | Reasoning | Grade school maths word problems | 30 |
| Custom adversarial | Safety | 16 harmful + legitimate prompt pairs | 16 |
| IFEval-style | Instruction Following | Format/length constraint tasks | 8 |
| Paraphrase pairs | Consistency | Same question, different phrasing | 8 |

---

## Architecture

```
Benchmark Datasets (TruthfulQA, GSM8K, custom)
        │
        ▼
Evaluation Engine
├── Unified model client (OpenAI + Groq APIs)
├── Dimension-specific scorers
│   ├── Factuality: multiple choice exact match
│   ├── Reasoning: final answer extraction + comparison
│   ├── Safety: refusal pattern detection
│   ├── Instruction following: constraint rule checking
│   └── Consistency: answer agreement across paraphrases
└── MLflow experiment tracking
        │
        ▼
SQLite Results Database
        │
        ▼
Plotly Dash Leaderboard
├── Medal ranking cards
├── Full dimension table
├── Radar chart (5 dimensions)
├── Grouped bar chart by dimension
└── Cost vs performance scatter
```

---

## Stack

OpenAI · Groq · HuggingFace Datasets · Plotly Dash · MLflow · pandas · scikit-learn

---

## Quickstart

```bash
git clone https://github.com/Shun024/llmbench.git
cd llmbench
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add API keys to .env
cp .env.example .env
# OPENAI_API_KEY=...
# GROQ_API_KEY=...  (free at console.groq.com)

# Run full benchmark (~15 min, ~$0.05)
PYTHONPATH=. python -m src.evaluation.runner

# Launch dashboard
PYTHONPATH=. python src/dashboard/app.py
# Open http://localhost:8053
```

---

## Adding New Models

Add any OpenAI-compatible or Groq-served model in `src/models/clients.py`:

```python
MODELS["mistral-7b"] = {
    "provider": "groq",
    "display_name": "Mistral 7B",
    "family": "Mistral AI",
}
```

---

## Adding New Benchmarks

Each benchmark implements two functions:
1. `load_*()` → returns `list[dict]` with `question`, `correct_answer`, `dimension`
2. `score_*()` → returns `dict` with `score` (0.0–1.0) and metadata

---

## Author

**Shun Le Yi Mon (Sheryl)** · Data Scientist · NLP & GenAI  
[LinkedIn](https://www.linkedin.com/in/shunleyimon724) · [GitHub](https://github.com/Shun024)
# Connect 4 LLM Benchmark

A deployable web app that benchmarks any LLM's Connect 4 move-selection skill
against an alpha-beta minimax oracle, with comparison against pre-baked
baselines (GPT-4o, Claude Sonnet 4.5, DeepSeek-Chat, and a human expert).

> **The deployable app lives in [`tool/`](tool/).** Start there.

```bash
cd tool
pip install -r requirements.txt
streamlit run app.py
```

See [`tool/README.md`](tool/README.md) for full setup, deployment instructions
(Streamlit Cloud / Docker / HF Spaces / Fly), and how to add new providers.

## Repository layout

```
.
├── tool/                       # Deployable Streamlit benchmarking app
│   ├── app.py                  #   UI
│   ├── core.py                 #   Connect 4 + minimax oracle
│   ├── llm_clients.py          #   Pluggable OpenAI-compat + Anthropic
│   ├── benchmark.py            #   Benchmark runner
│   ├── results_loader.py
│   ├── baseline_results/       #   Pre-baked: GPT-4o, Claude 4.5, DeepSeek, Human
│   ├── community_results/      #   Saved runs from app users (gitignored)
│   ├── generated_positions.json
│   ├── Dockerfile
│   └── requirements.txt
├── board.ipynb                 # Original CS152 final-project notebook
├── sumbit/                     # Original submission folder (results + scripts)
├── plots_and_stats.py          # Standalone plotting script (works on tool's JSON)
└── plot*.png                   # Pre-generated comparison plots
```

## Origin

This started as the CS152 final project (a Jupyter notebook benchmarking 3
LLM providers + a human expert across 8 categories of Connect 4 positions).
It's been refactored into a deployable web app so anyone can plug in their
own LLM API and add a row to the comparison.

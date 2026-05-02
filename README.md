# Connect 4 LLM Benchmark

A deployable web app that lets anyone plug in an LLM API and benchmark its
Connect 4 move-selection skill against a strong alpha-beta minimax oracle —
then compare against pre-baked baselines (GPT-4o, Claude Sonnet 4.5,
DeepSeek-Chat, and a human expert).

The benchmark covers **8 categories × 15 positions = 120 positions**:
forced wins (1/3/5 moves), forced blocks, mid-game, and late-game positions.
Every chosen move is graded by its rank distance from the minimax-optimal move.

## Quickstart (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the sidebar:
1. Pick a provider (OpenAI, Anthropic, DeepSeek, Groq, OpenRouter, Together,
   or any custom OpenAI-compatible endpoint).
2. Paste your API key (kept in-session only, never written to disk).
3. Pick a model and click **Run benchmark**. Start with 5 positions/category
   for a cheap smoke test, then scale up to the full 15.

Saved runs land in `community_results/` so they accumulate as comparison
baselines for future visitors.

## Deploy

`app.py`, `Dockerfile`, and `requirements.txt` are all at the repo root, so
no platform-specific root-directory override is needed.

### Streamlit Community Cloud (free, easiest)
1. <https://streamlit.io/cloud> → "New app"
2. Repo: this one. Branch: `main`. Main file: `app.py`.
3. **Don't** add API keys as secrets — users supply their own at runtime.

### Docker
```bash
docker build -t connect4-bench .
docker run -p 8501:8501 connect4-bench
```

### Hugging Face Spaces
Create a new Space with the **Streamlit** SDK, push these files, set `app.py`
as the entry point.

### Railway / Render / Fly.io
Any platform that runs the Dockerfile works. Expose port 8501.

## Repository layout

```
.
├── app.py                  # Streamlit UI (entry point)
├── core.py                 # Connect 4 board + alpha-beta minimax oracle
├── llm_clients.py          # Pluggable OpenAI-compatible + Anthropic transports
├── benchmark.py            # Benchmark runner with progress callbacks
├── results_loader.py       # Loads baseline + community runs
├── generated_positions.json
├── baseline_results/       # GPT-4o, Claude 4.5, DeepSeek, Human Expert
├── community_results/      # Saved runs from app users (gitignored)
├── Dockerfile
├── requirements.txt
├── .streamlit/config.toml
└── notebook/               # Original CS152 final-project notebook + plots
    ├── board.ipynb
    ├── plots_and_stats.py
    ├── plot*.png
    └── sumbit/             # Original submission folder (results JSONs)
```

## Adding a new provider

`llm_clients.py` exposes a `PROVIDER_PRESETS` dict. Add a new entry like:

```python
"MyProvider": {
    "transport": "openai_compatible",   # or "anthropic"
    "base_url": "https://api.example.com/v1",
    "default_model": "my-model-id",
    "key_env": "MY_PROVIDER_KEY",
},
```

Any OpenAI-compatible chat-completions endpoint works without code changes —
just use **Custom (OpenAI-compatible)** in the UI and paste the base URL.

## Security notes

- `.env` is gitignored. **Never commit API keys.**
- The deployed app accepts keys at runtime in the sidebar; they live only in
  the Streamlit session, not in logs or disk.
- If you fork from this repo, keys you put in your local `.env` stay local.

## How grading works

For each position, an iterative-deepening alpha-beta minimax (default depth 6,
configurable to 8) ranks every legal move. The LLM's chosen move gets a
**rank distance** — `0` if it picked the optimal move, `1` if second-best, etc.
Categories where there's a unique winning or blocking move (e.g. *P1 Win in 1*,
*P2 Win in 1 (Block)*) are the cleanest pass/fail tests; mid-game and late-game
categories show degrees of skill.

Result JSONs are byte-compatible with the original notebook output, so the
existing `notebook/plots_and_stats.py` analysis still works on community runs.

## Origin

This started as the CS152 final project (a Jupyter notebook benchmarking 3
LLM providers + a human expert across 8 categories of Connect 4 positions —
preserved at `notebook/board.ipynb`). It's been refactored into a deployable
web app so anyone can plug in their own LLM API and add a row to the
comparison.

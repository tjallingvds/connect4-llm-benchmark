"""Streamlit app: connect any LLM and benchmark it on Connect 4 against the
existing baselines (GPT-4o, Claude Sonnet 4.5, DeepSeek, Human Expert).

Run locally:
    streamlit run app.py
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from benchmark import load_positions, run_benchmark, save_results
from llm_clients import PROVIDER_PRESETS, build_mover
from results_loader import extract_metrics, load_baselines, load_community


HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(HERE, "generated_positions.json")
BASELINE_DIR = os.path.join(HERE, "baseline_results")
COMMUNITY_DIR = os.path.join(HERE, "community_results")

REASONING_BUCKETS = {
    "Immediate Action": ["P1 Win in 1", "P2 Win in 1 (Block)"],
    "Short Planning": ["Late Game", "P1 Win in 3", "P2 Win in 3"],
    "Long Planning": ["P1 Win in 5", "P2 Win in 5", "Early Game"],
}


def bucket_avg(metrics, cats):
    cat_to_acc = dict(zip(metrics["categories"], metrics["category_accuracies"]))
    vals = [cat_to_acc[c] for c in cats if c in cat_to_acc]
    return sum(vals) / len(vals) if vals else 0.0


st.set_page_config(page_title="Connect 4 LLM Benchmark", layout="wide")
st.title("Connect 4 LLM Benchmark")
st.caption(
    "Plug in any LLM API and benchmark it on 8 categories of Connect 4 positions, "
    "graded against an alpha-beta minimax oracle. Compare against GPT-4o, Claude "
    "Sonnet 4.5, DeepSeek, and a human expert baseline."
)


# ----- Sidebar: provider config -------------------------------------------------

st.sidebar.header("Connect your model")

provider = st.sidebar.selectbox("Provider", list(PROVIDER_PRESETS.keys()), index=0)
preset = PROVIDER_PRESETS[provider]

env_default_key = os.getenv(preset["key_env"], "") if preset.get("key_env") else ""
api_key = st.sidebar.text_input(
    "API key",
    value=env_default_key,
    type="password",
    help="Stored only in this Streamlit session — never written to disk.",
)

model = st.sidebar.text_input("Model ID", value=preset["default_model"])

base_url_default = preset.get("base_url") or ""
base_url = st.sidebar.text_input(
    "Base URL (OpenAI-compatible only)",
    value=base_url_default,
    help="Leave blank for default. Used for OpenAI-compatible providers.",
    disabled=(preset["transport"] != "openai_compatible"),
)

display_name = st.sidebar.text_input(
    "Display name (shown in charts)",
    value=model or "My Model",
)

st.sidebar.divider()
st.sidebar.subheader("Benchmark settings")

depth = st.sidebar.slider("Minimax depth (oracle)", 4, 8, 6,
                          help="Higher depth = stronger ground truth but slower scoring.")
max_per_cat = st.sidebar.slider(
    "Positions per category", 1, 15, 5,
    help="The full benchmark is 15 per category × 8 = 120 positions. Start small to control cost.",
)
save_run = st.sidebar.checkbox(
    "Save run to community_results/", value=True,
    help="Saved runs persist as comparison baselines for future visitors.",
)

run_btn = st.sidebar.button("Run benchmark", type="primary",
                            width="stretch",
                            disabled=not (api_key and model))


# ----- Load existing results ----------------------------------------------------

baselines = load_baselines(BASELINE_DIR)
community = load_community(COMMUNITY_DIR)

with st.expander("Existing baselines & community runs", expanded=True):
    rows = []
    for entry in baselines + community:
        m = entry["metrics"]
        rows.append({
            "Model": m["model"],
            "Kind": entry["kind"],
            "Overall": f"{m['overall_accuracy']:.1%}",
            "Immediate Action": f"{bucket_avg(m, REASONING_BUCKETS['Immediate Action']):.1%}",
            "Short Planning": f"{bucket_avg(m, REASONING_BUCKETS['Short Planning']):.1%}",
            "Long Planning": f"{bucket_avg(m, REASONING_BUCKETS['Long Planning']):.1%}",
            "Correct / total": f"{m['total_correct']}/{m['total_positions']}",
        })
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            "Reasoning buckets: **Immediate Action** = Win/Block in 1 · "
            "**Short Planning** = Late Game + Win in 3 · "
            "**Long Planning** = Win in 5 + Early Game."
        )
    else:
        st.info("No results loaded yet.")


# ----- Run -----------------------------------------------------------------------

if run_btn:
    try:
        positions_data = load_positions(POSITIONS_PATH)
    except FileNotFoundError:
        st.error(f"Could not find positions file at {POSITIONS_PATH}")
        st.stop()

    failure_log = []
    mover = build_mover(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url or None,
        on_failure=lambda err: failure_log.append(err),
    )

    progress = st.progress(0.0, text="Starting…")
    status = st.empty()
    live_table = st.empty()
    live_rows = []

    def _on_progress(done, total, label):
        progress.progress(done / total, text=f"{done}/{total} — {label}")

    def _on_position(category, idx, result):
        live_rows.append({
            "Category": category,
            "Position": idx,
            "Chose": result["chosen_move"],
            "Optimal": result["optimal_move"],
            "Rank distance": result["rank_distance"],
        })
        live_table.dataframe(live_rows[-12:], width="stretch", hide_index=True)

    started = time.time()
    with st.spinner(f"Benchmarking {display_name}…"):
        try:
            results = run_benchmark(
                get_moves=mover,
                positions_data=positions_data,
                depth=depth,
                max_per_category=max_per_cat,
                progress_cb=_on_progress,
                position_cb=_on_position,
            )
        except Exception as e:
            st.error(f"Benchmark crashed: {e}")
            st.stop()

    elapsed = time.time() - started
    progress.progress(1.0, text=f"Done in {elapsed:.1f}s")
    status.success(
        f"{display_name}: {results['total_correct']}/{results['total_positions']} "
        f"correct ({results['overall_accuracy']:.1%})"
    )
    if failure_log:
        with st.expander(f"⚠️ {len(failure_log)} API failures fell back"):
            for err in failure_log[:20]:
                st.code(err)

    # Save and reload comparison set
    saved_path = None
    if save_run:
        saved_path = save_results(
            results, model_name=display_name, out_dir=COMMUNITY_DIR,
            extra_meta={
                "provider": provider, "model": model,
                "base_url": base_url or preset.get("base_url") or "",
                "max_per_category": max_per_cat, "depth": depth,
            },
        )
        st.caption(f"Saved to `{os.path.relpath(saved_path, HERE)}`")
        community = load_community(COMMUNITY_DIR)

    new_metrics = extract_metrics(results, display_name)

    # ----- Comparison charts ---------------------------------------------------

    st.subheader("Comparison")

    all_metrics = [b["metrics"] for b in baselines] + [c["metrics"] for c in community]
    # Avoid double-counting the just-saved run if it landed in community
    seen = set()
    deduped = []
    for m in all_metrics:
        key = (m["model"], m["total_positions"], round(m["overall_accuracy"], 4))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(m)
    if not save_run:
        deduped.append(new_metrics)
    elif new_metrics["model"] not in {m["model"] for m in deduped}:
        deduped.append(new_metrics)

    # Overall accuracy bar chart
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [m["model"] for m in deduped]
        vals = [m["overall_accuracy"] for m in deduped]
        colors = ["#4c8cff" if m["model"] != display_name else "#ff6b35"
                  for m in deduped]
        bars = ax.barh(labels, vals, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Overall accuracy")
        ax.set_title("Overall accuracy")
        for bar, v in zip(bars, vals):
            ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.1%}", va="center", fontsize=9)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = [np.mean(m["avg_rank_distances"]) for m in deduped]
        colors = ["#4c8cff" if m["model"] != display_name else "#ff6b35"
                  for m in deduped]
        bars = ax.barh([m["model"] for m in deduped], vals, color=colors)
        ax.set_xlabel("Mean rank distance from optimal (lower is better)")
        ax.set_title("Mean rank distance")
        for bar, v in zip(bars, vals):
            ax.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{v:.2f}", va="center", fontsize=9)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)

    # Reasoning-complexity buckets
    st.markdown("**Performance by reasoning complexity**")
    st.caption(
        "Categories grouped by how much lookahead the position demands: "
        "**Immediate Action** (Win/Block in 1) · **Short Planning** "
        "(Late Game, Win in 3) · **Long Planning** (Win in 5, Early Game)."
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bucket_labels = list(REASONING_BUCKETS.keys())
    bx = np.arange(len(bucket_labels))
    bwidth = 0.8 / max(len(deduped), 1)
    for i, m in enumerate(deduped):
        offset = (i - (len(deduped) - 1) / 2) * bwidth
        vals = [bucket_avg(m, REASONING_BUCKETS[b]) for b in bucket_labels]
        color = "#ff6b35" if m["model"] == display_name else None
        bars = ax.bar(bx + offset, vals, bwidth, label=m["model"], color=color)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.0%}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(bx)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Average accuracy")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    st.pyplot(fig)

    # Per-category accuracy
    st.markdown("**Per-category accuracy**")
    fig, ax = plt.subplots(figsize=(11, 5))
    cats = new_metrics["categories"]
    x = np.arange(len(cats))
    width = 0.8 / max(len(deduped), 1)
    for i, m in enumerate(deduped):
        offset = (i - (len(deduped) - 1) / 2) * width
        color = "#ff6b35" if m["model"] == display_name else None
        ax.bar(x + offset, m["category_accuracies"], width,
               label=m["model"], color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    # Rationale samples
    st.markdown("**Sample rationales from this run**")
    samples = []
    for cat_name, cat in results["category_results"].items():
        for p in cat["positions"]:
            if p.get("rationale"):
                samples.append({
                    "Category": cat_name,
                    "Chose": p["chosen_move"],
                    "Optimal": p["optimal_move"],
                    "Rank dist": p["rank_distance"],
                    "Rationale": p["rationale"],
                })
        if len(samples) >= 12:
            break
    if samples:
        st.dataframe(samples[:12], width="stretch", hide_index=True)

    # Download
    import json as _json
    st.download_button(
        "Download full results JSON",
        data=_json.dumps({**results, "model_name": display_name}, indent=2),
        file_name=f"{display_name.replace(' ', '_')}_results.json",
        mime="application/json",
    )

else:
    st.info(
        "Configure a provider, paste your API key, and click **Run benchmark** "
        "in the sidebar. Tip: start with 5 positions per category for a quick "
        "smoke test, then scale up."
    )

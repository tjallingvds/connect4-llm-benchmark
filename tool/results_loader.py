"""Aggregate baseline + community results for cross-model comparison."""
import json
import os


BASELINE_LABELS = {
    "gpt-4o.json": "GPT-4o",
    "claude-sonnet-4-5.json": "Claude Sonnet 4.5",
    "deepseek-chat.json": "DeepSeek-Chat",
    "human-expert.json": "Human Expert",
}


def _load(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_metrics(results_dict, model_name):
    metrics = {
        "model": model_name,
        "overall_accuracy": results_dict["overall_accuracy"],
        "total_correct": results_dict["total_correct"],
        "total_positions": results_dict["total_positions"],
        "categories": [],
        "category_accuracies": [],
        "avg_rank_distances": [],
    }
    for cat_name, cat in results_dict["category_results"].items():
        metrics["categories"].append(cat_name)
        metrics["category_accuracies"].append(cat["accuracy"])
        rds = [p["rank_distance"] for p in cat["positions"]
               if p.get("rank_distance") is not None]
        metrics["avg_rank_distances"].append(sum(rds) / len(rds) if rds else 0)
    return metrics


def load_baselines(baseline_dir):
    out = []
    if not os.path.isdir(baseline_dir):
        return out
    for fname in sorted(os.listdir(baseline_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(baseline_dir, fname)
        try:
            data = _load(path)
        except Exception:
            continue
        label = BASELINE_LABELS.get(fname) or data.get("model_name") or fname.replace(".json", "")
        out.append({"label": label, "path": path, "data": data,
                    "metrics": extract_metrics(data, label), "kind": "baseline"})
    return out


def load_community(community_dir):
    out = []
    if not os.path.isdir(community_dir):
        return out
    for fname in sorted(os.listdir(community_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(community_dir, fname)
        try:
            data = _load(path)
        except Exception:
            continue
        label = data.get("model_name") or fname.replace(".json", "")
        out.append({"label": label, "path": path, "data": data,
                    "metrics": extract_metrics(data, label), "kind": "community"})
    return out

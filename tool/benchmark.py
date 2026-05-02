"""Benchmark runner. Mirrors the notebook's run_benchmark but yields progress.

The output dict is byte-compatible with the existing baseline JSONs
(baseline_results/*.json) so the same comparison code works for old and new runs.
"""
import json
import os
import time

from core import evaluate_position


CATEGORY_ORDER = [
    ("P1 Win in 1", "p1_win_in_1"),
    ("P2 Win in 1 (Block)", "p2_win_in_1_block"),
    ("P1 Win in 3", "p1_win_in_3"),
    ("P2 Win in 3", "p2_win_in_3"),
    ("P1 Win in 5", "p1_win_in_5"),
    ("P2 Win in 5", "p2_win_in_5"),
    ("Early Game", "early_after_5_moves"),
    ("Late Game", "late_after_20_moves"),
]


def load_positions(path):
    with open(path, "r") as f:
        return json.load(f)


def build_position_sets(positions_data, max_per_category=None):
    sets = []
    for label, key in CATEGORY_ORDER:
        positions = positions_data[key]
        if max_per_category is not None:
            positions = positions[:max_per_category]
        sets.append((label, positions, 1))
    return sets


def run_benchmark(get_moves, positions_data, depth=6, max_per_category=None,
                  progress_cb=None, position_cb=None):
    """Run the benchmark.

    Args:
        get_moves: function(board, player, valid_moves) -> int or {move, rationale}
        positions_data: dict loaded from generated_positions.json
        depth: minimax search depth used as ground truth
        max_per_category: optional cap (cost control / quick smoke tests)
        progress_cb: optional fn(done, total, label) called after each position
        position_cb: optional fn(category, index, result_dict) for live UI updates

    Returns:
        results dict in the same shape as the baseline JSON files.
    """
    sets = build_position_sets(positions_data, max_per_category=max_per_category)
    total = sum(len(positions) for _, positions, _ in sets)
    done = 0
    started = time.time()

    all_results = {}
    for label, positions, player in sets:
        category_results = []
        for i, board in enumerate(positions):
            result = evaluate_position(board, player, depth, get_moves)
            clean = {
                "position_index": i,
                "category": label,
                "chosen_move": result["chosen_move"],
                "optimal_move": result["best_move"],
                "rank_distance": (result["ranking"] - 1
                                  if result["ranking"] is not None else None),
                "raw_rankings": result["rankings"],
                "raw_scores": result["scores"],
            }
            if result.get("error"):
                clean["error"] = result["error"]
            if result.get("rationale"):
                clean["rationale"] = result["rationale"]
            category_results.append(clean)

            done += 1
            if position_cb:
                position_cb(label, i, clean)
            if progress_cb:
                progress_cb(done, total, label)

        correct = sum(1 for r in category_results if r.get("rank_distance") == 0)
        accuracy = correct / len(category_results) if category_results else 0.0
        all_results[label] = {
            "correct": correct,
            "total": len(category_results),
            "accuracy": accuracy,
            "positions": category_results,
        }

    total_correct = sum(c["correct"] for c in all_results.values())
    total_positions = sum(c["total"] for c in all_results.values())
    overall_accuracy = total_correct / total_positions if total_positions else 0.0

    return {
        "total_correct": total_correct,
        "total_positions": total_positions,
        "overall_accuracy": overall_accuracy,
        "category_results": all_results,
        "elapsed_seconds": round(time.time() - started, 2),
    }


def save_results(results, model_name, out_dir, extra_meta=None):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_name)
    if not safe_name:
        safe_name = "model"
    payload = dict(results)
    payload["model_name"] = model_name
    payload["timestamp"] = int(time.time())
    if extra_meta:
        payload["meta"] = extra_meta
    path = os.path.join(out_dir, f"{safe_name}_{payload['timestamp']}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path

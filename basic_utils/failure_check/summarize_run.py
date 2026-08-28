"""
Summarize an evaluation run from its episodes.jsonl stream.

The record.txt written during a run only holds running averages, so per-episode
outcomes cannot be recovered from it. EpisodeRecorder appends one structured
line per episode instead; this script turns that stream into a breakdown by
result category and lists the false positives with the commands to reproduce
them.

Usage:
    python -m basic_utils.failure_check.summarize_run videos/test_hm3dv2_val
    python -m basic_utils.failure_check.summarize_run videos/test_hm3dv2_val \
        --category "false positive" --category stucking
"""

import argparse
import json
import os
import shlex

from prettytable import PrettyTable

from params import RESULT_TYPES, result_dirname


def load_episodes(summary_path):
    """Read episodes.jsonl, keeping the last entry per run_index.

    A resumed run replays the episode it died on, so the same run_index can
    appear more than once; the later entry is the one that completed.
    """
    if not os.path.exists(summary_path):
        raise SystemExit(f"No summary file at {summary_path}")

    by_index = {}
    malformed = 0
    with open(summary_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            key = entry.get("run_index")
            by_index[key] = entry

    if malformed:
        print(f"Skipped {malformed} malformed line(s) in {summary_path}")

    return [by_index[k] for k in sorted(by_index, key=lambda v: (v is None, v))]


def summarize(episodes):
    """Aggregate metrics and per-category counts."""
    total = len(episodes)
    if total == 0:
        return None

    def _mean(key):
        values = [e[key] for e in episodes if e.get(key) is not None]
        return sum(values) / len(values) if values else 0.0

    counts = {}
    for entry in episodes:
        counts[entry.get("result", "unknown")] = (
            counts.get(entry.get("result", "unknown"), 0) + 1
        )

    return {
        "total": total,
        "success_rate": 100.0 * _mean("success"),
        "spl": 100.0 * _mean("spl"),
        "soft_spl": 100.0 * _mean("soft_spl"),
        "distance_to_goal": _mean("distance_to_goal"),
        "counts": counts,
    }


EXPL_RESULT_NAMES = {
    0: "EXPLORATION", 1: "SEARCH_BEST_OBJECT", 2: "SEARCH_OVER_DEPTH_OBJECT",
    3: "SEARCH_SUSPICIOUS_OBJECT", 4: "NO_PASSABLE_FRONTIER",
    5: "NO_COVERABLE_FRONTIER", 6: "SEARCH_EXTREME",
}


def commit_point(record_dir):
    """Find the step at which the planner committed to an object.

    Two independent signals, because they can disagree and the disagreement is
    itself informative:

      * the FSM switching out of EXPLORE into SEARCH_OBJECT, and which branch
        of planNextBestPoint produced it (a genuine confidence pass vs. one of
        the suspicious/extreme fallbacks)
      * the first object cluster whose fused confidence cleared the acceptance
        gate, with the confidence and observation count at that moment
    """
    steps_path = os.path.join(record_dir, "steps.jsonl")
    if not os.path.exists(steps_path):
        return None

    fsm_commit = None
    gate_commit = None
    with open(steps_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            if fsm_commit is None and entry.get("final_state") == 1:
                fsm_commit = {
                    "step": entry.get("step"),
                    "expl_result": entry.get("expl_result"),
                }

            fusion = entry.get("object_fusion")
            if gate_commit is None and fusion:
                for cluster in fusion.get("clusters", []):
                    if cluster.get("is_confident") and cluster.get("best_label") == 0:
                        gate_commit = {
                            "step": entry.get("step"),
                            "cluster_id": cluster.get("id"),
                            "confidence": cluster.get("confidence"),
                            "threshold": fusion.get("min_confidence"),
                            "observations": cluster.get("observation_num"),
                            "min_observations": fusion.get("min_observation_num"),
                            "centroid": cluster.get("centroid"),
                        }
                        break

    if fsm_commit is None and gate_commit is None:
        return None
    return {"fsm": fsm_commit, "gate": gate_commit}


def print_commit_point(record_dir, indent="    "):
    """Print the commitment analysis for one record, if the data is there."""
    commit = commit_point(record_dir)
    if commit is None:
        return

    gate = commit.get("gate")
    if gate:
        print(
            f"{indent}gate crossed at step {gate['step']}: cluster {gate['cluster_id']} "
            f"confidence {gate['confidence']:.3f} >= {gate['threshold']:.3f} "
            f"after {gate['observations']} observations "
            f"(min {gate['min_observations']}) at "
            f"({gate['centroid'][0]:.2f}, {gate['centroid'][1]:.2f})"
        )
    fsm = commit.get("fsm")
    if fsm:
        branch = EXPL_RESULT_NAMES.get(fsm["expl_result"], fsm["expl_result"])
        print(f"{indent}FSM committed at step {fsm['step']} via {branch}")
        if branch in ("SEARCH_SUSPICIOUS_OBJECT", "SEARCH_EXTREME"):
            print(
                f"{indent}  note: committed through a low-confidence fallback, "
                "not the confidence gate"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", help="run output directory, e.g. videos/test_hm3dv2_val"
    )
    parser.add_argument(
        "--summary-file", default="episodes.jsonl", help="name of the summary stream"
    )
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="result category to list in detail (repeatable, default: false positive)",
    )
    parser.add_argument(
        "--dataset", default="hm3dv2", help="dataset name used in replay commands"
    )
    args = parser.parse_args()

    categories = args.category or ["false positive"]

    episodes = load_episodes(os.path.join(args.run_dir, args.summary_file))
    stats = summarize(episodes)
    if stats is None:
        raise SystemExit("Summary file is empty - no episodes recorded yet.")

    metrics = PrettyTable(["Metric", "Value"])
    metrics.add_row(["Episodes", stats["total"]])
    metrics.add_row(["Success rate", f"{stats['success_rate']:.2f}%"])
    metrics.add_row(["SPL", f"{stats['spl']:.2f}%"])
    metrics.add_row(["Soft SPL", f"{stats['soft_spl']:.2f}%"])
    metrics.add_row(["Avg distance to goal", f"{stats['distance_to_goal']:.4f}"])
    print(metrics)

    breakdown = PrettyTable(["Result", "Count", "Share"])
    known = [r for r in RESULT_TYPES if r in stats["counts"]]
    extra = sorted(set(stats["counts"]) - set(RESULT_TYPES))
    for result in known + extra:
        count = stats["counts"][result]
        breakdown.add_row([result, count, f"{100.0 * count / stats['total']:.1f}%"])
    print(breakdown)

    arms = {
        json.dumps((e.get("planner") or {}), sort_keys=True) for e in episodes
    }
    if len(arms) > 1:
        print(
            f"\nWARNING: {len(arms)} different planner fusion configurations appear "
            "in this run - the aggregate above mixes ablation arms."
        )
    for arm in sorted(arms):
        print(f"Planner config: {arm}")

    for category in categories:
        matches = [e for e in episodes if e.get("result") == category]
        print(f"\n=== {category} ({len(matches)}) ===")
        if not matches:
            continue
        for entry in matches:
            record_dir = os.path.join(
                args.run_dir,
                "records",
                result_dirname(category),
                "epi{run}_{scene}_{eid}_{label}".format(
                    run=entry.get("run_index"),
                    scene=entry.get("scene"),
                    eid=entry.get("episode_id"),
                    label=_slug(entry.get("target_label", "object")),
                ),
            )
            exists = "" if os.path.isdir(record_dir) else "  [artifacts not written]"
            print(
                f"\n  run_index={entry.get('run_index')}  "
                f"scene={entry.get('scene')}  episode_id={entry.get('episode_id')}  "
                f"target={entry.get('target_label')}"
            )
            print(
                f"    steps={entry.get('count_steps')}  "
                f"d(goal)={entry.get('distance_to_goal')}  "
                f"final_state={entry.get('final_state')}  "
                f"expl_result={entry.get('expl_result')}"
            )
            print(f"    record: {record_dir}{exists}")
            print_commit_point(record_dir)
            print(
                f"    replay live:    python habitat_evaluation.py "
                f"--dataset {args.dataset} test_epi_num={entry.get('run_index')}"
            )
            # result categories contain spaces, so the path needs quoting
            print(
                f"    replay offline: python -m basic_utils.record_episode."
                f"replay_episode {shlex.quote(record_dir)}"
            )


def _slug(label):
    primary = str(label).split("|")[0].strip().lower()
    slug = "".join(c if c.isalnum() else "_" for c in primary).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "object"


if __name__ == "__main__":
    main()

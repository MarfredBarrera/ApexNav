"""Replay a recorded gate episode offline against a perception server (no Habitat, no planner).

    python -m perception_gate.replay --episode-dir videos/sync_d1/gate/epi574 \\
        --url http://3dawareverification-dev-1:8771 --out videos/d1_mv_replay/epi574 \\
        [--set multiview=true multiview_grid=true ...] [--pin-initial] [--dry-run]

The episode's ``records/<outcome>/<name>/`` (``record.flush_on=[all]``: steps.jsonl, rgb/%06d.jpg
for every step, masks/<step>_<k>.png) is fed to a fresh ``VerificationGate`` step by step
(``observe``), and at every step where the original run made a decision (verification/
decisions.jsonl) the same request is sent to ``handle``: approach re-requests carry the original
approach id, viewpoint status, agent position and step count. The planner is not re-run, so the
trajectory after the first decision that differs from the original is counterfactual: the replay
stops there. Inputs are the recorded JPEG frames (the live gate sent PNG of the raw frame), so a
replayed verdict may differ slightly from the original even with the same settings.

Gate settings: the dataset's habitat_eval yaml ``verification`` section, with ``enabled``,
``approach`` (true), ``on_error`` (reject), ``url`` and ``log_dir`` (``--out``) set here, then
``--set key=value`` (YAML values, e.g. ``multiview=true multiview_variant=forced2``).
``--pin-initial``: requests that are not approach re-requests reuse the original verdict
(no server call), so a run with other first-request settings (e.g. ``--d1-evidence input``)
still reaches the same approaches and only the close-range decisions are re-judged.
``--dry-run``: no server at all: first requests reuse the original verdict, /verify_views
answers ``--dry-run-status`` (to check the wiring on CPU).

Writes ``<out>/decisions.jsonl`` (the gate's log, with views/ images), ``<out>/replay.jsonl``
(one line per decision, original vs replayed) and ``<out>/summary.json``; prints a table.
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import yaml

from perception_gate import client
from perception_gate.gate import ARRIVED, STATUS_NAMES, VerificationGate

STATUS_CODES = {name: code for code, name in STATUS_NAMES.items()}
STATE_NAMES = {1: "ACCEPTED", -1: "REJECTED", 2: "DEFERRED", 3: "APPROACH", 0: "UNVERIFIED"}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_episode(episode_dir):
    """(record folder, meta, steps, original decisions) of a recorded gate episode."""
    found = sorted(glob.glob(os.path.join(episode_dir, "records", "*", "*", "steps.jsonl")))
    if len(found) != 1:
        raise FileNotFoundError(f"{episode_dir}: expected one records/<outcome>/<name>/steps.jsonl, found {found}")
    folder = os.path.dirname(found[0])
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as stream:
        meta = json.load(stream)
    with open(found[0], encoding="utf-8") as stream:
        steps = [json.loads(line) for line in stream if line.strip()]
    path = os.path.join(episode_dir, "verification", "decisions.jsonl")
    decisions = []
    if os.path.exists(path):  # no file: the gate made no decision in this episode
        with open(path, encoding="utf-8") as stream:
            decisions = [json.loads(line) for line in stream if line.strip()]
    if meta.get("dropped_frames"):
        raise ValueError(f"{folder}: {meta['dropped_frames']} dropped frames; rgb/%06d.jpg is not the step")
    return folder, meta, steps, decisions


def gate_settings(dataset, url, out, overrides=()):
    """The verification settings of ``config/habitat_eval_<dataset>.yaml`` for a d1 replay, the
    episode step limit and the camera intrinsics."""
    with open(os.path.join(REPO, "config", f"habitat_eval_{dataset}.yaml"), encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    cfg = dict(config.get("verification") or {})
    cfg.update(enabled=True, approach=True, on_error="reject", url=url, log_dir=out)
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, not {override!r}")
        cfg[key.strip().removeprefix("+").removeprefix("verification.")] = yaml.safe_load(value)
    habitat = config.get("habitat") or {}
    max_steps = (habitat.get("environment") or {}).get("max_episode_steps", 500)
    sensor = (((habitat.get("simulator") or {}).get("agents") or {}).get("main_agent") or {}) \
        .get("sim_sensors", {}).get("rgb_sensor") or {}
    camera = {key: sensor.get(key) for key in ("hfov", "width", "height", "position")} if sensor else None
    return cfg, int(max_steps), camera


def step_inputs(folder, record):
    """The ``observe`` arguments of one recorded step."""
    step = int(record["step"])
    bgr = cv2.imread(os.path.join(folder, "rgb", f"{step:06d}.jpg"), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"{folder}/rgb/{step:06d}.jpg")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    masks, labels, scores, centroids, instances = [], [], [], [], []
    for detection in record.get("detections") or []:
        mask = None
        if detection.get("mask"):
            mask = cv2.imread(os.path.join(folder, "masks", detection["mask"]), cv2.IMREAD_GRAYSCALE)
        masks.append(np.zeros(rgb.shape[:2], np.uint8) if mask is None else (mask > 0).astype(np.uint8))
        labels.append(int(detection["label_idx"]))
        scores.append(float(detection["score"]))
        centroids.append(detection.get("world_centroid"))
        instances.append(detection.get("semantic_instances"))
    return (step, rgb, masks, labels, scores, centroids, instances, record["pose"], float(record["camera_pitch"]),
            record.get("object_fusion"))


def request_of(decision):
    """The VerificationRequest of an original decision line."""
    cluster = decision["cluster"]
    request = {"seq": decision.get("seq", 0), "cluster_id": cluster["id"], "centroid_x": cluster["centroid"][0],
               "centroid_y": cluster["centroid"][1], "confidence": cluster.get("confidence", 0.0),
               "observation_num": cluster.get("observation_num", 0), "expl_result": decision.get("expl_result", -1),
               "at_stop": decision.get("at_stop", False), "approach_id": 0}
    approach = decision.get("approach") or {}
    if "status" in approach:  # a re-request (an issued approach has "issued" instead)
        request.update(approach_id=approach["id"], viewpoint_status=STATUS_CODES.get(approach["status"], 0),
                       viewpoint_x=approach["agent"][0], viewpoint_y=approach["agent"][1],
                       approach_steps=approach.get("steps", 0))
    return request


def kind_of(decision):
    approach = decision.get("approach") or {}
    return approach.get("status", "stop" if decision.get("at_stop") else "initial")


class Replayer:
    """Feeds one recorded episode to a gate; ``verify``/``verify_views`` reach the server unless pinned."""

    def __init__(self, episode_dir, url, out, overrides=(), pin_initial=False, dry_run=False,
                 dry_run_status="insufficient", dataset=None, verify=None, verify_views=None, ping=None):
        self.folder, self.meta, self.steps, self.decisions = load_episode(episode_dir)
        self.episode_dir, self.out = episode_dir, out
        self.pin_initial, self.dry_run, self.dry_run_status = pin_initial or dry_run, dry_run, dry_run_status
        dataset = dataset or self.meta.get("dataset", "hm3dv2")
        cfg, self.max_steps, camera = gate_settings(dataset, url, out, overrides)
        self.current = None  # the original decision being replayed
        self._server_verify = verify or client.verify
        self._server_views = verify_views or client.verify_views
        if dry_run:
            ping = ping or (lambda url, timeout: {"ok": True, "settings": {
                "selector": "d1", "multiview": True, "multiview_grid": True, "dry_run": True}})
        os.makedirs(out, exist_ok=True)
        self.gate = VerificationGate(cfg, out, verify=self._verify, verify_views=self._verify_views, ping=ping,
                                     camera=camera)
        goals = [g.get("object_id") for g in self.meta.get("ground_truth_goals") or []]
        self.gate.start_episode(self.meta["run_index"], self.meta.get("scene_id", self.meta.get("scene", "")),
                                self.meta.get("episode_id", ""), self.meta["target_label"], goal_object_ids=goals,
                                max_episode_steps=self.max_steps)

    def _pinned(self):
        return self.pin_initial and self.current is not None and kind_of(self.current) in ("initial", "stop")

    def _verify(self, url, rgb, mask, target, **kwargs):
        if self._pinned() or self.dry_run:
            verdict = self.current.get("verdict")
            if verdict is None:  # the original request failed
                raise RuntimeError(f"original request failed: {self.current.get('error')}")
            return {"ok": True, "verdict": verdict, "view_image": None, "pinned": True}
        return self._server_verify(url, rgb, mask, target, **kwargs)

    def _verify_views(self, url, views, target, **kwargs):
        if self.dry_run:
            per_view = [{"step": v["step"], "index": v["index"], "scores": v.get("scores"), "label": None,
                         "margin": None, "cached": v.get("scores") is not None} for v in views]
            return {"ok": True, "grid_image": None, "verdict": {
                "status": self.dry_run_status, "source": "qwen", "accepted": self.dry_run_status == "correct",
                "qwen": {"status": self.dry_run_status, "dry_run": True}, "per_view": per_view,
                "aggregate": None, "n_views": len(views)}}
        return self._server_views(url, views, target, **kwargs)

    def run(self, log=print):
        """Replays until the first decision that differs; returns the replay lines."""
        by_step = {}
        for decision in self.decisions:
            by_step.setdefault(int(decision["step"]), []).append(decision)
        last = max(by_step) if by_step else -1
        lines, diverged = [], False
        for record in self.steps:
            if diverged or int(record["step"]) > last:
                break
            inputs = step_inputs(self.folder, record)
            self.gate.observe(*inputs)
            for decision in by_step.get(int(record["step"]), []):
                self.current = decision
                start = time.time()
                state = self.gate.handle(request_of(decision), inputs[-1], int(record["step"]))
                new = self._last_logged()
                line = {"episode": self.meta["run_index"], "step": decision["step"], "seq": decision.get("seq"),
                        "cluster": decision["cluster"]["id"], "kind": kind_of(decision),
                        "old_state": decision.get("state"), "new_state": state,
                        "old_policy": decision.get("policy"), "new_policy": new.get("policy"),
                        "old_status": decision.get("status"), "new_status": new.get("status"),
                        "old_model": decision.get("model"), "new_model": new.get("model"),
                        "pinned": self._pinned(), "n_views": len(new.get("views") or []) or None,
                        "aggregate": (new.get("verdict") or {}).get("aggregate") if new.get("views") else None,
                        "qwen": _qwen_brief((new.get("verdict") or {}).get("qwen")),
                        "ground_truth": (new.get("ground_truth") or {}).get("is_correct_instance"),
                        "error": new.get("error"), "seconds": round(time.time() - start, 2),
                        "same": state == decision.get("state")}
                lines.append(line)
                log(f"epi{line['episode']} s{line['step']:>3} c{line['cluster']:<3} {line['kind']:<11} "
                    f"{STATE_NAMES.get(line['old_state'], line['old_state']):>9} -> "
                    f"{STATE_NAMES.get(state, state):<9} {'' if line['same'] else 'DIFF '}"
                    f"{line['new_policy'] or line['new_model'] or ''}")
                if not line["same"]:
                    diverged = True  # the planner would have done something else from here on
                    break
        self.current = None
        with open(os.path.join(self.out, "replay.jsonl"), "w", encoding="utf-8") as stream:
            for line in lines:
                stream.write(json.dumps(line) + "\n")
        summary = {"episode_dir": os.path.abspath(self.episode_dir), "run_index": self.meta["run_index"],
                   "target": self.meta["target_label"], "result": self.meta.get("result"),
                   "decisions": len(self.decisions), "replayed": len(lines),
                   "first_diff": next((l for l in lines if not l["same"]), None),
                   "final_old": lines[-1]["old_state"] if lines else None,
                   "final_new": lines[-1]["new_state"] if lines else None,
                   "counts": self.gate.finish_episode(), "settings": {k: v for k, v in self.gate.cfg.items()
                                                                      if k.startswith(("multiview", "approach"))},
                   "pin_initial": self.pin_initial, "dry_run": self.dry_run}
        with open(os.path.join(self.out, "summary.json"), "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=1)
        return lines

    def _last_logged(self):
        path = os.path.join(self.out, "decisions.jsonl")
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as stream:
            lines = stream.read().splitlines()
        return json.loads(lines[-1]) if lines else {}


def _qwen_brief(qwen):
    if not qwen:
        return None
    return {k: qwen.get(k) for k in ("status", "probabilities", "error", "reconstruct_error", "grid", "n_tokens",
                                     "latency_s", "prompt_version", "forced2_p_correct") if k in qwen}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m perception_gate.replay", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode-dir", required=True, nargs="+",
                        help="videos/<run>/gate/epiN (several: --out is then a parent folder)")
    parser.add_argument("--url", default="http://3dawareverification-dev-1:8771")
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="verification settings")
    parser.add_argument("--dataset", help="habitat_eval_<dataset>.yaml (default: the episode's meta.json)")
    parser.add_argument("--pin-initial", action="store_true",
                        help="first requests reuse the original verdict (only approach re-requests reach the server)")
    parser.add_argument("--dry-run", action="store_true", help="no server (wiring check)")
    parser.add_argument("--dry-run-status", default="insufficient", choices=("correct", "wrong", "insufficient"))
    args = parser.parse_args(argv)
    many = len(args.episode_dir) > 1
    failed = 0
    for episode_dir in args.episode_dir:
        out = os.path.join(args.out, os.path.basename(os.path.normpath(episode_dir))) if many else args.out
        if os.path.exists(os.path.join(out, "summary.json")):
            print(f"{out}: done, skipped")
            continue
        if os.path.exists(os.path.join(out, "decisions.jsonl")):  # a killed attempt: start afresh
            os.rename(out, f"{out}.partial.{time.strftime('%Y%m%d%H%M%S')}")
        try:
            Replayer(episode_dir, args.url, out, args.set, args.pin_initial, args.dry_run, args.dry_run_status,
                     args.dataset).run()
        except Exception as error:  # one broken episode does not stop the others
            failed += 1
            print(f"{episode_dir}: {type(error).__name__}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

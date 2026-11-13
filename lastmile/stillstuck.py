"""Start states at the FIRST "Still stuck at the same place" planner event after the gate accept.

planner.log has no step markers: step = (number of `PUB_ACTION to WAIT_ACTION_FINISH` lines before the event) + offset,
offset = accept_step - (same count at the first `[Verification] cluster N ACCEPTED` line); offset is 0 for 85% of episodes
(gate-driven approach steps publish no native action, so it can be > 0). Sanity: the agent must be stationary around the mapped step.
Writes bench_data/episodes_stillstuck.tsv and episodes/stillstuck_<set>_<ep>.npz (all tags that have the event; SUCCESS ones are the
regression check).      python -m lastmile.stillstuck
"""
import csv
import glob
import json
import sys

import numpy as np

from lastmile import bench

V = "videos/"
RUNS = {"test2": "test2_nolm/gate", "heldout": "heldout_final/gate", "tuning": "tuning_final/gate", "fresh400": "fresh400_final/gate"}


def event_step(d, accept_step):
    pubs, acc, ev = 0, None, None
    for l in open(d + "/planner.log", errors="ignore"):
        if "PUB_ACTION to WAIT_ACTION_FINISH" in l:
            pubs += 1
        elif acc is None and "[Verification] cluster" in l and "ACCEPTED" in l:
            acc = pubs
        elif acc is not None and ev is None and "Still stuck at the same place" in l:
            ev = pubs
            break
    if acc is None or ev is None:
        return None, acc
    return ev + (accept_step - acc), acc


def main():
    import habitat_sim
    import quaternion
    sys.path.insert(0, "scripts")
    from sample_episodes import iterator_episodes
    eps, _ = iterator_episodes("hm3dv2")
    rows = [r for r in csv.DictReader(open(f"{bench.BENCH}/episodes.tsv"), delimiter="\t")]
    plan = []
    for r in rows:
        d = f"{V}{RUNS[r['set']]}/epi{r['ep']}"
        s, acc = event_step(d, int(r["accept_step"]))
        if s is not None:
            plan.append((r, d, s))
    out, sim, scene = [], None, None
    for r, d, s in sorted(plan, key=lambda x: x[0]["scene"]):
        if r["scene"] != scene:
            sim and sim.close()
            sim, scene = bench.make_sim(r["scene"]), r["scene"]
        steps = [json.loads(l) for l in open(glob.glob(d + "/records/*/*/steps.jsonl")[0]) if l.strip()]
        if s > steps[-1]["step"] or s < int(r["accept_step"]):
            print("SKIP", r["set"], r["ep"], "mapped step", s, "outside", steps[-1]["step"])
            continue
        X = eps[int(r["ep"])]
        st = habitat_sim.AgentState()
        st.position, st.rotation = np.array(X.start_position, np.float32), np.quaternion(X.start_rotation[3], *X.start_rotation[:3])
        sim.get_agent(0).set_state(st)
        trail = {}
        for t in steps:
            if t["action"] in (1, 2, 3):
                sim.get_agent(0).act(("move_forward", "turn_left", "turn_right")[t["action"] - 1])
            if t["step"] >= s - 1 and t["step"] <= s + 3:
                trail[t["step"]] = bench.pose(sim)[0]
            if t["step"] == s:
                pos, yaw = bench.pose(sim)
                pitch = t["camera_pitch"]
        # stationary check over steps s..s+3 (the planner reports no progress there)
        still = max(float(np.hypot(*(p - pos)[[0, 2]])) for p in trail.values())
        z = np.load(f"{bench.BENCH}/episodes/{r['set']}_{r['ep']}.npz")
        d_gt, dy = bench.score_point(pos, z["gt_points"])
        out.append(dict(set=r["set"], ep=r["ep"], tag=r["tag"], scene=r["scene"], target=r["target"], accept_step=r["accept_step"],
                        stillstuck_step=s, remaining_steps=500 - s, final_step=steps[-1]["step"], moved_within_3_steps_m=round(still, 3),
                        start_x=round(pos[0], 3), start_y=round(pos[1], 3), start_z=round(pos[2], 3), start_yaw=round(yaw, 4),
                        start_pitch=round(pitch, 3), dist_to_gt=round(d_gt, 2), native_bench=r["native_bench"]))
        np.savez_compressed(f"{bench.BENCH}/episodes/stillstuck_{r['set']}_{r['ep']}.npz", position=pos, yaw=yaw, pitch=pitch,
                            goal_cloud=z["goal_cloud"], gt_points=z["gt_points"])
    sim and sim.close()
    out.sort(key=lambda o: (o["tag"], o["set"], int(o["ep"])))
    cols = list(out[0].keys())
    with open(f"{bench.BENCH}/episodes_stillstuck.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for o in out:
            f.write("\t".join(str(o[c]) for c in cols) + "\n")
    for o in out:
        print(o["tag"], o["set"], o["ep"], "step", o["stillstuck_step"], "of", o["final_step"], "budget", o["remaining_steps"], "moved", o["moved_within_3_steps_m"], "d_gt", o["dist_to_gt"])


if __name__ == "__main__":
    main()

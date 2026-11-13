"""Build the 'stuck' start states: for every FAIL episode, the agent pose where the native episode ended
(last step of records/*/steps.jsonl). Rescue-eligible = native final_state == 2 (STUCKING, params.FINAL_RESULT).
Writes bench_data/episodes_stuck.tsv and bench_data/episodes/stuck_<set>_<ep>.npz (inside the container).

    python -m lastmile.stuck
"""
import csv
import glob
import json
import sys

import numpy as np

from lastmile import bench

V = "videos/"
RUNS = {"test2": "test2_nolm/gate", "heldout": "heldout_final/gate", "tuning": "tuning_final/gate", "fresh400": "fresh400_final/gate"}
STATES = {0: "EXPLORE", 1: "SEARCH_OBJECT", 2: "STUCKING", 3: "NO_FRONTIER", 4: "REACH_OBJECT"}


def main():
    import habitat_sim
    import quaternion
    sys.path.insert(0, "scripts")
    from sample_episodes import iterator_episodes
    eps, _ = iterator_episodes("hm3dv2")
    rows = [r for r in csv.DictReader(open(f"{bench.BENCH}/episodes.tsv"), delimiter="\t") if r["tag"] == "FAIL" and (r["set"], r["ep"]) != ("test2", "701")]   # 701: stopped 0.5 m from the bed, retagged SUCCESS
    out, sim, scene = [], None, None
    for r in sorted(rows, key=lambda r: r["scene"]):
        if r["scene"] != scene:
            sim and sim.close()
            sim, scene = bench.make_sim(r["scene"]), r["scene"]
        d = f"{V}{RUNS[r['set']]}/epi{r['ep']}"
        E = json.loads(open(d + "/episodes.jsonl").readline())
        steps = [json.loads(l) for l in open(glob.glob(d + "/records/*/*/steps.jsonl")[0]) if l.strip()]
        X = eps[int(r["ep"])]
        st = habitat_sim.AgentState()
        st.position, st.rotation = np.array(X.start_position, np.float32), np.quaternion(X.start_rotation[3], *X.start_rotation[:3])
        sim.get_agent(0).set_state(st)
        for s in steps:
            if s["action"] in (1, 2, 3):
                sim.get_agent(0).act(("move_forward", "turn_left", "turn_right")[s["action"] - 1])
        pos, yaw = bench.pose(sim)
        z = np.load(f"{bench.BENCH}/episodes/{r['set']}_{r['ep']}.npz")
        d_gt, dy = bench.score_point(pos, z["gt_points"])
        last = steps[-1]["step"]
        fs = int(E["final_state"])
        out.append(dict(set=r["set"], ep=r["ep"], tag=r["tag"], scene=r["scene"], target=r["target"], native_final_state=f"{fs}:{STATES.get(fs)}",
                        native_end=E.get("result", ""), stuck_step=last, remaining_steps=500 - last, rescue_eligible=int(fs == 2),
                        start_x=round(pos[0], 3), start_y=round(pos[1], 3), start_z=round(pos[2], 3), start_yaw=round(yaw, 4),
                        start_pitch=round(steps[-1]["camera_pitch"], 3), dist_to_gt=round(d_gt, 2), accept_step=r["accept_step"]))
        np.savez_compressed(f"{bench.BENCH}/episodes/stuck_{r['set']}_{r['ep']}.npz", position=pos, yaw=yaw, pitch=steps[-1]["camera_pitch"],
                            goal_cloud=z["goal_cloud"], gt_points=z["gt_points"])
    sim and sim.close()
    out.sort(key=lambda o: (o["set"], int(o["ep"])))
    cols = list(out[0].keys())
    with open(f"{bench.BENCH}/episodes_stuck.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for o in out:
            f.write("\t".join(str(o[c]) for c in cols) + "\n")
    for o in out:
        print(o["set"], o["ep"], o["native_final_state"], "stuck_step", o["stuck_step"], "budget", o["remaining_steps"], "d_gt", o["dist_to_gt"], "eligible", o["rescue_eligible"])


if __name__ == "__main__":
    main()

"""Merge oracle results into episodes.tsv and write README.md (+ publish to the outputs dir). Plain python."""
import collections
import csv
import os
import shutil

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_data")
OUT = os.path.expanduser("~/codes/3DAwareVerification/outputs/apexnav_online/lastmile_bench_2026-11-11")


def rd(p):
    return list(csv.DictReader(open(p), delimiter="\t"))


def status(r):
    if int(r["success"]):
        return "ok"
    if int(r["oracle_cands"]) == 0:
        return "no navigable point within 1 m of the object on this floor"
    if r["oracle_geo"] == "":
        return "no path (other island/floor)"
    if int(r["oracle_reach_exhausted"]):
        return "kinematically trapped (reachable set under 0.25 m/30 deg actions never gets within 1 m)"
    return "budget too small / search cap"


RETAG = {("test2", "701"): "SUCCESS"}      # corrected 1 m scorer: stopped 0.5 m from the bed
eps = rd(f"{B}/episodes.tsv")
RUNS = {"std": "oracle_{b}", "fine10": "oracle_fine10_{b}", "fine125": "oracle_fine125_{b}"}
res = {n: {b: {(r["set"], r["ep"]): r for r in rd(f"{B}/results/{p.format(b=b)}.tsv")} for b in ("remaining", "150")}
       for n, p in RUNS.items() if os.path.exists(f"{B}/results/{p.format(b='150')}.tsv")}
newcols = [f"oracle_{n}_{b}" for n in res for b in ("remaining", "150")] + ["oracle_actions", "oracle_status"] + \
          [f"oracle_{n}_actions" for n in res if n != "std"]
cols = [c for c in eps[0].keys() if c not in newcols and c not in ("oracle_remaining", "oracle_150")] + newcols
for e in eps:
    k = (e["set"], e["ep"])
    if "orig_tag" not in e or not e.get("orig_tag"):
        e["orig_tag"] = e["tag"]
    e["tag"] = "offfloor" if e.get("floor_ok") == "0" else RETAG.get(k, e["orig_tag"])
    e["grp"] = ("fresh400-" if e["set"] == "fresh400" else "") + e["tag"]
    e["failure_class"] = "exploration/other" if e["tag"] == "offfloor" else ("last-mile" if e["tag"] == "FAIL" else "")
    for n in res:
        for b in ("remaining", "150"):
            e[f"oracle_{n}_{b}"] = res[n][b][k]["success"]
    e["oracle_actions"] = res["std"]["remaining"][k]["actions"]
    e["oracle_status"] = status(res["std"]["remaining"][k])
    for n in res:
        if n != "std":
            e[f"oracle_{n}_actions"] = res[n]["remaining"][k]["actions"]
for c in ("orig_tag", "failure_class"):
    if c not in cols:
        cols.append(c)
with open(f"{B}/episodes.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for e in eps:
        f.write("\t".join(e.get(c, "") for c in cols) + "\n")
tab = ["| tag | n | " + " | ".join(f"{n} rem / 150 / mean actions" for n in res) + " |", "|---|---|" + "---|" * len(res)]
for t in ("FAIL", "SUCCESS", "SUCCESS_capstop", "benchonly_1m", "budget", "offfloor", "fresh400-FAIL", "fresh400-SUCCESS", "fresh400-offfloor"):
    s = [e for e in eps if e["grp"] == t]
    if not s:
        continue
    cells = []
    for n in res:
        ok = [e for e in s if e[f"oracle_{n}_remaining"] == "1"]
        acts = [int(res[n]["remaining"][(e["set"], e["ep"])]["actions"]) for e in ok]
        cells.append(f"{len(ok)} / {sum(e[f'oracle_{n}_150'] == '1' for e in s)} / {sum(acts) / len(acts) if acts else 0:.1f}")
    tab.append(f"| {t}{' (not last-mile: exploration/other)' if t.endswith('offfloor') else ' (TEST ONLY split)' if t.startswith('fresh400') else ''} | {len(s)} | " + " | ".join(cells) + " |")
fails = ["| set | ep | tag | " + " | ".join(res) + " | std oracle status |", "|---|---|---|" + "---|" * len(res) + "---|"] + [
    f"| {e['set']} | {e['ep']} | {e['tag']} | " + " | ".join(e[f"oracle_{n}_remaining"] for n in res) + f" | {e['oracle_status']} |"
    for e in eps if any(e[f"oracle_{n}_remaining"] != "1" for n in res)]
txt = open(f"{os.path.dirname(B)}/README_template.md").read()
txt += "\n## Oracle reference (A* over the real simulator dynamics to a navigable cell within 0.99 m of the GT object, then STOP)\n\n"
txt += "std = ApexNav actions (0.25 m, 30 deg); fine10 / fine125 = std + FORWARD_SMALL 0.10 / 0.125 m + 15 deg turns (`--actions fine --small-m`). Cells: successes with the remaining budget / with cap 150 / mean actions of the successes.\n\n"
txt += "\n".join(tab) + "\n\nEpisodes some oracle cannot solve (feasibility limits; `1` = solved):\n\n"
txt += "\n".join(fails) + "\n\nNative-logged reference = `native_bench` / `native_eye` columns (no sim run): the logged ApexNav pipeline outcome; `baseline_eye` = baseline run.\n"
txt += "\nFloor rule: `floor_ok` / `n_targets_same_floor` (see `floors.py`); episodes whose targets are all off the agent's floor get tag `offfloor` (original tag in `orig_tag`) and `failure_class = exploration/other`: not navigator-recoverable, so left out of the last-mile FAIL headline (but still counted in overall SR tables, never dropped).\n"
open(f"{B}/README.md", "w").write(txt)
os.makedirs(OUT, exist_ok=True)
for n in ("episodes.tsv", "README.md"):
    shutil.copy(f"{B}/{n}", OUT)
for d in ("episodes", "results"):
    shutil.copytree(f"{B}/{d}", f"{OUT}/{d}", dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.log"))
print("\n".join(tab))
print("\n".join(fails))

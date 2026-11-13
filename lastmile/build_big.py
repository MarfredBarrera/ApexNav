"""Big bench: every post-accept episode of the 600 tuning eps (test2/heldout/tuning) that the old pipeline got S under the
manual 1 m rule (success1m_2026-11-11/manual1m.tsv, pipeline arm) and whose accepted cluster is not flagged wrong (is_correct_instance False dropped; None = unlogged, kept),
plus the existing post-accept FAILs. Writes lastmile/bench_data_big (LASTMILE_BENCH must point there). Never fresh400.
    LASTMILE_BENCH=lastmile/bench_data_big python -m lastmile.build_big
"""
import collections
import csv
import os

from lastmile import build

MAN = os.path.expanduser("~/codes/3DAwareVerification/outputs/apexnav_online/success1m_2026-11-11/manual1m.tsv")
if not os.path.exists(MAN):
    MAN = "lastmile/bench_data/src/success1m.manual1m.tsv"
SET = {"held": "heldout", "test2": "test2", "tune": "tuning"}


def select(cands):
    sel = collections.OrderedDict()
    for r in csv.DictReader(open(f"lastmile/bench_data/src/{build.LASTMILE_TSV}"), delimiter="\t"):
        key = (r["set"], int(r["episode"]))
        if r["class"].startswith("A_") and key in cands:
            sel[key] = ("FAIL", r["class"], r["note"])
    nS = dropped = 0
    for r in csv.DictReader(open(MAN), delimiter="\t"):
        if r["arm"] != "pipeline" or r["label_1m"] != "S":
            continue
        key = (SET[r["set"]], int(r["episode"]))
        if key in sel:
            continue
        if key not in cands or cands[key]["correct"] is False:
            dropped += 1
            continue
        sel[key] = ("SUCCESS", "S", "")
        nS += 1
    print("big bench: S", nS, "FAIL", sum(v[0] == "FAIL" for v in sel.values()), "old-S without accept/correct cluster", dropped)
    return sel


build.RUNS.pop("fresh400")
build.select = select
if __name__ == "__main__":
    build.main()

"""Per-subset comparison table of navigators from accept (plain python).  python3 lastmile/analyze.py hybrid_v1"""
import collections
import csv
import sys

B = "/home/marfred/codes/ApexNav/lastmile/bench_data"
rd = lambda p: {(r["set"], r["ep"]): r for r in csv.DictReader(open(f"{B}/results/{p}.tsv"), delimiter="\t")}
hy = sys.argv[1]
M = {(r["set"], r["ep"]): r for r in csv.DictReader(open(f"{B}/episodes.tsv"), delimiter="\t")}
H = rd(hy)
D = {**rd("ddppo_success_accept"), **rd("ddppo_fail_accept")}
A, OS, OF = rd("astar_remaining"), rd("oracle_remaining"), rd("oracle_fine10_remaining")
tag = lambda k: "SUCCESS" if M[k]["tag"] == "SUCCESS" and k[0] != "fresh400" else M[k]["tag"]
def cls(r):
    if r["success"] == "1":
        return "success"
    if r["stopped"] == "1":
        return "stop>1m"
    return "collision-loop" if int(r["collisions"]) > 0.4 * max(1, int(r["actions"])) else "wander"
ks = [k for k in H]
print("| subset | n | hybrid succ/forced | ddppo succ/forced | astar succ | oracle std | oracle fine10 |\n|---|---|---|---|---|---|---|")
for t in ("FAIL", "SUCCESS", "SUCCESS_capstop"):
    s = [k for k in ks if tag(k) == t]
    g = lambda R, col="success": sum(int(R[k][col]) for k in s if k in R)
    print(f"| {t} | {len(s)} | {g(H)}/{g(H, 'success_forced')} | {g(D)}/{g(D, 'success_forced')} | {g(A)} | {g(OS)} | {g(OF)} |")
fail = [k for k in ks if tag(k) == "FAIL"]
succ = [k for k in ks if tag(k) in ("SUCCESS", "SUCCESS_capstop")]
for name, R in (("hybrid", H), ("ddppo", D), ("astar", A)):
    resc = sum(int(R[k]["success"]) for k in fail if k in R)
    lost = sum(1 - int(R[k]["success"]) for k in succ if k in R)
    print(f"{name}: rescued FAIL {resc} - lost SUCCESS(+capstop) {lost} = net {resc - lost}   (n_fail {len(fail)}, n_succ {len(succ)})")
for name, R in (("hybrid", H), ("ddppo", D)):
    for grp, kk in (("FAIL", fail), ("SUCC", succ)):
        print(name, grp, dict(collections.Counter(cls(R[k]) for k in kk if k in R)))
print("lost SUCCESS (hybrid):", [(k, tag(k), cls(H[k]), H[k]["final_dist"]) for k in succ if H[k]["success"] == "0"])
print("per-ep vs oracle std (FAIL): hybrid wins/losses")
for k in fail:
    h, o, f = H[k]["success"], OS[k]["success"], OF[k]["success"]
    flag = "BEATS-ORACLE" if h == "1" and o == "0" and f == "0" else ("win" if h == "1" else "loss")
    print(k, "hyb", h, "ddppo", D[k]["success"], "astar", A.get(k, {}).get("success", "-"), "orc", o, "fine", f, "bud", H[k]["budget"], cls(H[k]), H[k]["final_dist"], flag)

"""Disable transparent huge pages in every Python process in the container.

Python imports this at startup because docker/apexnav-env.sh puts the workspace
root on PYTHONPATH in both use_core and use_vlm. On the shared host no NUMA node
has free 2 MB blocks, so a huge-page fault stalls in direct compaction (seconds
per MB). The stall counts as memory pressure, and systemd-oomd then SIGKILLs a
whole container (this one or the perception server's). numpy madvises huge pages
for large arrays by default. PR_SET_THP_DISABLE is inherited by child processes,
so the C++ nodes started by `ros2 launch` are covered too.
"""
import ctypes
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
PR_SET_THP_DISABLE = 41
if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
    print(f"sitecustomize: PR_SET_THP_DISABLE failed (errno {ctypes.get_errno()})", flush=True)

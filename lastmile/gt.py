"""GT semantic-instance points of HM3D scenes (scoring only; navigators never see these).
Parsing adapted from 3DAwareVerification/outputs/apexnav_online/success1m_2026-11-11/onem_full.py."""
import json
import struct

import numpy as np

SCENES = "data/scene_datasets/hm3d_v0.2"
TOKENS = {"chair": {"chair"}, "bed": {"bed"}, "sofa": {"sofa", "couch"}, "plant": {"plant"},
          "toilet": {"toilet"}, "tv_monitor": {"tv", "television", "monitor"}}


def _glb(glb):
    b = open(glb, "rb").read()
    off, js, bin_ = 12, None, None
    while off < len(b):
        n, t = struct.unpack("<II", b[off:off + 8])
        chunk = b[off + 8:off + 8 + n]
        if t == 0x4E4F534A:
            js = json.loads(chunk)
        elif t == 0x004E4942:
            bin_ = chunk
        off += 8 + n
    return js, bin_


def load_objects(glb, txt):
    js, bin_ = _glb(glb)
    comp = {5126: np.float32, 5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5120: np.int8, 5122: np.int16}
    nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def acc(i):
        a = js["accessors"][i]
        bv = js["bufferViews"][a["bufferView"]]
        dt = np.dtype(comp[a["componentType"]])
        n = nc[a["type"]]
        stride = bv.get("byteStride", dt.itemsize * n)
        o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        raw = np.frombuffer(bin_, dtype=np.uint8, count=a["count"] * stride, offset=o)
        raw = raw.reshape(a["count"], stride)[:, :dt.itemsize * n].copy().view(dt).reshape(a["count"], n)
        return raw, a

    def mat(nd):
        if "matrix" in nd:
            return np.array(nd["matrix"]).reshape(4, 4).T
        M = np.eye(4)
        if "scale" in nd:
            M = np.diag(list(nd["scale"]) + [1]) @ M
        if "rotation" in nd:
            x, y, z, w = nd["rotation"]
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            M2 = np.eye(4)
            M2[:3, :3] = R
            M = M2 @ M
        if "translation" in nd:
            M3 = np.eye(4)
            M3[:3, 3] = nd["translation"]
            M = M3 @ M
        return M

    pts, cols = [], []

    def walk(ni, P):
        nd = js["nodes"][ni]
        M = P @ mat(nd)
        if "mesh" in nd:
            for pr in js["meshes"][nd["mesh"]]["primitives"]:
                at = pr["attributes"]
                if "COLOR_0" not in at:
                    continue
                p, _ = acc(at["POSITION"])
                c, a = acc(at["COLOR_0"])
                c = c[:, :3].astype(float)
                c = c / 65535.0 if a["componentType"] == 5123 else c / 255.0 if a["componentType"] == 5121 else c
                pts.append((M[:3, :3] @ p.T.astype(float)).T + M[:3, 3])
                cols.append(c)
        for ch in nd.get("children", []):
            walk(ch, M)

    for r in js["scenes"][js.get("scene", 0)]["nodes"]:
        walk(r, np.eye(4))
    P, C = np.concatenate(pts), np.concatenate(cols)
    objs = {}
    for line in open(txt).read().splitlines()[1:]:
        parts = line.split(",")
        objs[int(parts[0])] = (int(parts[1], 16), parts[2].strip('"'))
    return P, C, objs


def _l2s(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


class SceneObjects:
    """instance id -> (N,3) habitat-world points; names; category lookup."""

    def __init__(self, scene_dir, short):
        dd = f"{SCENES}/val/{scene_dir}/{short}"
        P, C, self.objs = load_objects(dd + ".semantic.glb", dd + ".semantic.txt")
        c8 = np.round(np.clip(_l2s(C), 0, 1) * 255).astype(int)
        self.K = c8[:, 0] * 65536 + c8[:, 1] * 256 + c8[:, 2]
        self.P = np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1)
        self.cache = {}

    def points(self, oid):
        oid = int(oid)
        if oid not in self.cache:
            self.cache[oid] = self.P[self.K == self.objs[oid][0]].astype(np.float32)
        return self.cache[oid]

    def category_ids(self, cat):
        toks = TOKENS[cat]
        return [o for o, (_, nm) in self.objs.items() if toks & set(nm.lower().replace("-", " ").replace("_", " ").split())]

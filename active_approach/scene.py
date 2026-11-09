"""Habitat scene, camera and ground-truth object geometry for the active-approach tests.

The camera matches ApexNav's evaluation embodiment (config/habitat_eval_hm3dv2.yaml):
640x480, hfov 79, 0.88 m high, radius 0.18, forward 0.25 m, turns 30 deg, no sliding. The runner's
actions are move_forward / turn_left / turn_right only, at a fixed camera pitch of 0 (ApexNav's pitch
after its initial look-down spin); look_up/look_down exist only to re-create logged starts.

GT visibility of an object comes from its own surface (faces of the HM3D semantic
mesh whose semantic texture colour is the object's), projected into the current
camera and tested against the rendered depth and semantic images.
"""
import glob
import io
import json
import math
import os
import struct

import numpy as np

HFOV, WIDTH, HEIGHT, CAMERA_HEIGHT = 79.0, 640, 480, 0.88
FORWARD_M, TURN_DEG, AGENT_RADIUS = 0.25, 30.0, 0.18
DEPTH_MAX = 20.0          # GT depth, metres (ApexNav's own depth stops at 5 m)
SCENES = "data/scene_datasets/hm3d_v0.2"
SCENE_CONFIG = f"{SCENES}/hm3d_annotated_basis.scene_dataset_config.json"
FOCAL = (WIDTH / 2) / math.tan(math.radians(HFOV) / 2)


def scene_path(scene):
    """'4ok3usBNeis' -> data/scene_datasets/hm3d_v0.2/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"""
    hits = glob.glob(f"{SCENES}/*/*-{scene}/{scene}.basis.glb")
    if not hits:
        raise FileNotFoundError(f"no HM3D-v2 scene {scene} under {SCENES}")
    return hits[0]


def make_sim(scene, gpu=0):
    import habitat_sim
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = SCENE_CONFIG
    sim_cfg.scene_id = scene_path(scene)
    sim_cfg.gpu_device_id = gpu
    sim_cfg.enable_physics = False
    sim_cfg.allow_sliding = False   # ApexNav: habitat_sim_v0.allow_sliding false (a blocked step does not slide)
    sensors = []
    for uuid, kind in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH),
                       ("semantic", habitat_sim.SensorType.SEMANTIC)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = uuid, kind
        spec.resolution = [HEIGHT, WIDTH]
        spec.position = [0.0, CAMERA_HEIGHT, 0.0]
        spec.hfov = HFOV
        if kind == habitat_sim.SensorType.DEPTH:
            spec.far = DEPTH_MAX
        sensors.append(spec)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.radius, agent.height = AGENT_RADIUS, CAMERA_HEIGHT
    agent.sensor_specifications = sensors
    act, spec = habitat_sim.agent.ActionSpec, habitat_sim.agent.ActuationSpec
    agent.action_space = {
        "move_forward": act("move_forward", spec(amount=FORWARD_M)),
        "turn_left": act("turn_left", spec(amount=TURN_DEG)),
        "turn_right": act("turn_right", spec(amount=TURN_DEG)),
        "look_up": act("look_up", spec(amount=TURN_DEG)),
        "look_down": act("look_down", spec(amount=TURN_DEG)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent]))
    return sim


def yaw_quat(yaw):
    """Rotation about +y by ``yaw`` (rad); yaw 0 looks along -z."""
    import quaternion  # numpy-quaternion, a habitat_sim dependency
    return quaternion.from_rotation_vector([0.0, yaw, 0.0])


def set_pose(sim, position, yaw, pitch=0.0):
    """Place the agent at ``position`` facing ``yaw``; camera ``pitch`` (rad, negative = down) in
    30-degree look steps as in ApexNav. Returns the observations."""
    import habitat_sim
    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = yaw_quat(yaw)
    agent.set_state(state, reset_sensors=True)
    for _ in range(int(round(abs(math.degrees(pitch)) / TURN_DEG))):
        agent.act("look_down" if pitch < 0 else "look_up")
    return sim.get_sensor_observations()


def agent_pose(sim):
    """(position xyz, yaw rad, camera pitch rad) of agent 0."""
    import quaternion
    state = sim.get_agent(0).get_state()
    forward = quaternion.rotate_vectors(state.rotation, np.array([0.0, 0.0, -1.0]))
    yaw = math.atan2(-forward[0], -forward[2])
    cam = state.sensor_states["rgb"]
    look = quaternion.rotate_vectors(cam.rotation, np.array([0.0, 0.0, -1.0]))
    pitch = math.asin(max(-1.0, min(1.0, look[1])))
    return np.asarray(state.position, dtype=float), yaw, pitch


def camera(sim):
    """(position, 3x3 rotation camera->world) of the rgb sensor."""
    import quaternion
    cam = sim.get_agent(0).get_state().sensor_states["rgb"]
    return np.asarray(cam.position, dtype=float), quaternion.as_rotation_matrix(cam.rotation)


def project(points, cam_pos, cam_rot):
    """World points -> (u, v, z) pixel coords and depth (z > 0 in front)."""
    local = (np.asarray(points) - cam_pos) @ cam_rot   # rows: R^T (p - t)
    z = -local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = WIDTH / 2 + FOCAL * local[:, 0] / z
        v = HEIGHT / 2 - FOCAL * local[:, 1] / z
    return u, v, z


def backproject(depth, mask, cam_pos, cam_rot, stride=2):
    """World points of the masked depth pixels (subsampled by ``stride``)."""
    vv, uu = np.nonzero(mask[::stride, ::stride])
    vv, uu = vv * stride, uu * stride
    z = depth[vv, uu]
    keep = (z > 0.05) & (z < DEPTH_MAX - 0.1)
    vv, uu, z = vv[keep], uu[keep], z[keep]
    local = np.stack([(uu - WIDTH / 2) * z / FOCAL, -(vv - HEIGHT / 2) * z / FOCAL, -z], 1)
    return local @ cam_rot.T + cam_pos


# ---------------------------------------------------------------- GT object surfaces
def _read_glb(path):
    with open(path, "rb") as f:
        f.read(12)
        length, _ = struct.unpack("<I4s", f.read(8))
        meta = json.loads(f.read(length))
        length, _ = struct.unpack("<I4s", f.read(8))
        blob = f.read(length)
    return meta, blob


def _accessor(meta, blob, index):
    a = meta["accessors"][index]
    view = meta["bufferViews"][a["bufferView"]]
    dtype = {5126: np.float32, 5121: np.uint8, 5123: np.uint16, 5125: np.uint32}[a["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
    return np.frombuffer(blob, dtype, a["count"] * width, offset).reshape(a["count"], width)


def semantic_objects(scene):
    """{id: name} from <scene>.semantic.txt."""
    path = scene_path(scene).replace(".basis.glb", ".semantic.txt")
    out = {}
    for line in open(path).read().splitlines()[1:]:
        parts = line.split(",")
        out[int(parts[0])] = parts[2].strip('"').strip().lower()
    return out


def object_faces(scene, cache_dir):
    """Every semantic-mesh face as (centroid, normal, area, object id), Habitat frame.

    Object id per face = semantic.txt id of the semantic texture colour at the face's
    centroid UV. Cached as <cache_dir>/<scene>_faces.npz (~10 MB per scene).
    """
    cache = os.path.join(cache_dir, f"{scene}_faces.npz")
    if os.path.exists(cache):
        data = np.load(cache)
        return data["centroid"], data["normal"], data["area"], data["id"]
    from PIL import Image
    glb = scene_path(scene).replace(".basis.glb", ".semantic.glb")
    meta, blob = _read_glb(glb)
    colour_to_id = {}
    for line in open(glb.replace(".glb", ".txt")).read().splitlines()[1:]:
        parts = line.split(",")
        colour_to_id[int(parts[1], 16)] = int(parts[0])
    lut_keys = np.array(sorted(colour_to_id), dtype=np.int64)
    lut_ids = np.array([colour_to_id[k] for k in lut_keys], dtype=np.int32)
    by_material = {}
    for mesh in meta["meshes"]:
        for prim in mesh["primitives"]:
            by_material.setdefault(prim["material"], []).append(prim)
    cs, ns, areas, ids = [], [], [], []
    for material, prims in by_material.items():
        tex = meta["materials"][material]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        view = meta["bufferViews"][meta["images"][meta["textures"][tex]["source"]]["bufferView"]]
        start = view.get("byteOffset", 0)
        image = np.asarray(Image.open(io.BytesIO(blob[start:start + view["byteLength"]])).convert("RGB"))
        h, w, _ = image.shape
        for prim in prims:
            pos = _accessor(meta, blob, prim["attributes"]["POSITION"]).astype(np.float64)
            uv = _accessor(meta, blob, prim["attributes"]["TEXCOORD_0"])
            tri = (_accessor(meta, blob, prim["indices"]).reshape(-1, 3) if "indices" in prim
                   else np.arange(len(pos)).reshape(-1, 3))
            p = pos[tri]
            mid_uv = uv[tri].mean(1)
            x = np.clip((mid_uv[:, 0] % 1) * w, 0, w - 1).astype(int)
            y = np.clip((mid_uv[:, 1] % 1) * h, 0, h - 1).astype(int)
            c = image[y, x].astype(np.int64)
            key = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
            slot = np.clip(np.searchsorted(lut_keys, key), 0, len(lut_keys) - 1)
            cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            area = np.linalg.norm(cross, axis=1) / 2
            cs.append(p.mean(1))
            ns.append(cross / np.maximum(2 * area[:, None], 1e-12))
            areas.append(area)
            ids.append(np.where(lut_keys[slot] == key, lut_ids[slot], 0))
    to_habitat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)  # glTF z-up -> y-up
    centroid = (np.concatenate(cs) @ to_habitat.T).astype(np.float32)
    normal = (np.concatenate(ns) @ to_habitat.T).astype(np.float32)
    area, oid = np.concatenate(areas).astype(np.float32), np.concatenate(ids).astype(np.int32)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache, centroid=centroid, normal=normal, area=area, id=oid)
    return centroid, normal, area, oid


class ObjectSurface:
    """Area-weighted surface samples of one object (points + outward normals)."""

    def __init__(self, centroid, normal, area, n=3000, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(area), size=min(n, len(area)), replace=len(area) < n,
                         p=area / area.sum()) if len(area) else np.zeros(0, int)
        self.points, self.normals = centroid[idx].astype(float), normal[idx].astype(float)
        self.center = (centroid * area[:, None]).sum(0) / max(area.sum(), 1e-9) if len(area) else None
        self.bottom = float(centroid[:, 1].min()) if len(area) else None
        self.size = (centroid.max(0) - centroid.min(0)) if len(area) else None


def surfaces(scene, cache_dir, ids):
    """{id: ObjectSurface} for the requested object ids."""
    centroid, normal, area, oid = object_faces(scene, cache_dir)
    return {i: ObjectSurface(centroid[oid == i], normal[oid == i], area[oid == i], seed=int(i))
            for i in ids}


def visibility(obs, surface, object_id, cam_pos, cam_rot):
    """GT visibility of one object in one frame.

    Over the object's camera-facing surface samples (minus self-occluded ones):
    visible = in frame and the first surface along the ray; truncated = outside the
    frame or behind the camera; occluded = another surface is in front.
    Also the visible pixel count/area fraction, distance, and mean |cos| between the
    visible surface normals and the viewing rays (low = oblique).
    """
    sem = np.asarray(obs["semantic"]).squeeze()
    depth = np.asarray(obs["depth"]).squeeze()
    pixels = int((sem == object_id).sum())
    out = {"pixels": pixels, "area": pixels / float(WIDTH * HEIGHT), "vis_frac": 0.0,
           "trunc_frac": 0.0, "occ_frac": 0.0, "facing_cos": None, "distance": None}
    if surface.center is None:
        return out
    flat = np.array([surface.center[0] - cam_pos[0], surface.center[2] - cam_pos[2]])
    out["distance"] = float(np.hypot(*flat))
    rays = surface.points - cam_pos
    facing = (surface.normals * -rays).sum(1) > 0
    if not facing.any():
        return out
    pts, ray = surface.points[facing], rays[facing]
    u, v, z = project(pts, cam_pos, cam_rot)
    inside = (z > 0.05) & (u >= 0) & (u < WIDTH) & (v >= 0) & (v < HEIGHT)
    ui, vi = np.clip(u, 0, WIDTH - 1).astype(int), np.clip(v, 0, HEIGHT - 1).astype(int)
    first = inside & (depth[vi, ui] >= z - (0.05 + 0.02 * z))
    own = inside & (sem[vi, ui] == object_id)
    visible = first & own
    self_occluded = inside & ~first & own
    denom = int(facing.sum() - self_occluded.sum())
    if denom <= 0:
        return out
    out["vis_frac"] = float(visible.sum() / denom)
    out["trunc_frac"] = float((~inside).sum() / denom)
    out["occ_frac"] = float((inside & ~visible & ~self_occluded).sum() / denom)
    if visible.any():
        n = surface.normals[facing][visible]
        r = ray[visible] / np.linalg.norm(ray[visible], axis=1, keepdims=True)
        out["facing_cos"] = float(np.abs((n * r).sum(1)).mean())
    return out

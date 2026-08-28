"""
Small helpers shared by the evaluation loop and the offline replay.

Deliberately free of ROS and habitat imports so that
basic_utils/record_episode/replay_episode.py can use the same pose convention
without pulling in the planner's message packages.
"""

import gzip
import json

import numpy as np

from vlm.Labels import MP3D_ID_TO_NAME


def slugify_label(label):
    """Turn a target category into a filename-safe token.

    Categories may list alternatives and contain spaces, e.g.
    "table | dining table | coffee table | desk" -> "table".
    """
    primary = str(label).split("|")[0].strip().lower()
    slug = "".join(c if c.isalnum() else "_" for c in primary).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "object"


def agent_world_pose(observations):
    """Agent pose in the planner's world frame as (x, y, yaw).

    Mirrors the habitat -> ROS convention used by ROSPublisherNonNode, so the
    pose lines up with everything the planner publishes.
    """
    gps = observations["gps"]
    compass = observations["compass"]
    yaw = float(np.asarray(compass).reshape(-1)[0])
    return float(-gps[2]), float(-gps[0]), yaw


def episode_goal_positions_in_planner_frame(episode):
    """Project Habitat goal positions into the planner's episodic XY frame.

    Habitat's GPS coordinates are the world displacement from the start pose,
    rotated by the inverse start quaternion.  The planner uses that same GPS
    frame as ``(-gps.z, -gps.x)``.  Applying the complete transform here lets
    a recorded object-map cluster be compared with a ground-truth goal for map
    debugging.  It is not used to decide object identity; semantic instance
    IDs under the committed detector mask make that decision exactly.
    """
    start = np.asarray(episode.start_position, dtype=np.float64)
    quat = np.asarray(episode.start_rotation, dtype=np.float64)
    if start.shape != (3,) or quat.shape != (4,):
        return []
    quat /= np.linalg.norm(quat)
    qvec, qw = quat[:3], quat[3]
    inverse_qvec = -qvec

    def rotate_into_start_frame(vector):
        # Optimized q * v * q^-1 for the inverse unit quaternion.
        return (
            2.0 * np.dot(inverse_qvec, vector) * inverse_qvec
            + (qw * qw - np.dot(inverse_qvec, inverse_qvec)) * vector
            + 2.0 * qw * np.cross(inverse_qvec, vector)
        )

    goals = []
    for index, goal in enumerate(getattr(episode, "goals", ())):
        position = getattr(goal, "position", None)
        if position is None:
            continue
        local = rotate_into_start_frame(
            np.asarray(position, dtype=np.float64) - start
        )
        goal_info = {
            "goal_index": index,
            "planner_position": [float(-local[2]), float(-local[0])],
            "habitat_position": [float(value) for value in position],
        }
        object_id = getattr(goal, "object_id", None)
        if object_id is not None:
            goal_info["object_id"] = str(object_id)
        goals.append(goal_info)
    return goals


def semantic_instances_for_masks(semantic, masks, max_instances=None):
    """Summarize Habitat semantic instance IDs beneath each detector mask.

    ``HabitatSimSemanticSensor`` uses scene object IDs, which are the same
    identifiers stored by ObjectNav goals.  Only per-instance pixel counts are
    retained; the full semantic image is unnecessary for forensic replay.  By
    default every visible ID is kept so the goal-overlap calculation is exact.
    """
    if semantic is None:
        return [None] * len(masks or [])
    semantic = np.asarray(semantic).squeeze()
    if semantic.ndim != 2:
        return [None] * len(masks or [])

    summaries = []
    for mask in masks or []:
        binary = np.asarray(mask).squeeze().astype(bool)
        if binary.shape != semantic.shape or not binary.any():
            summaries.append(None)
            continue
        values = semantic[binary]
        values = values[values > 0]
        if not len(values):
            summaries.append([])
            continue
        ids, counts = np.unique(values, return_counts=True)
        order = np.argsort(counts)[::-1]
        if max_instances is not None:
            order = order[:max_instances]
        summaries.append([
            {"id": str(int(ids[index])), "pixels": int(counts[index])}
            for index in order
        ])
    return summaries


def semantic_goal_overlap(instances, goal_object_ids, min_overlap=0.5):
    """Compare one detector-mask semantic summary against ObjectNav goals.

    The evidence is deliberately pixel-weighted: an incidental sliver of a
    goal object inside a loose detector mask must not turn a wrong-object
    commitment into a correct-instance commitment.  ``instances`` is the
    compact structure returned by :func:`semantic_instances_for_masks`.
    """
    if instances is None:
        return None
    goal_ids = {str(object_id) for object_id in goal_object_ids if object_id is not None}
    observed = [
        {
            "id": str(instance["id"]),
            "pixels": int(instance["pixels"]),
        }
        for instance in instances
        if "id" in instance and "pixels" in instance
    ]
    total_pixels = sum(instance["pixels"] for instance in observed)
    matching_pixels = sum(
        instance["pixels"] for instance in observed if instance["id"] in goal_ids
    )
    overlap = matching_pixels / total_pixels if total_pixels else 0.0
    return {
        "source": "semantic_mask",
        "goal_object_ids": sorted(goal_ids),
        "observed_instances": observed,
        "semantic_pixels": total_pixels,
        "goal_pixels": matching_pixels,
        "goal_overlap_fraction": round(overlap, 4),
        "match_threshold": float(min_overlap),
        "is_correct_instance": bool(goal_ids) and overlap >= min_overlap,
    }


def load_category_mapping(
    val_path="data/datasets/objectnav/mp3d/v1/val/val.json.gz",
):
    """Map habitat object categories onto the detector's label names.

    HM3D and MP3D episodes name their targets with MP3D category ids; the
    detectors expect COCO-style names. Returns (category_to_coco, id_to_name).
    """
    with gzip.open(val_path, "rt", encoding="utf-8") as handle:
        val_data = json.load(handle)

    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }
    return category_to_coco, id_to_name


def resolve_label(label, category_to_coco, id_to_name):
    """Convert an episode's object_category into the detector's label name."""
    if label in category_to_coco:
        return id_to_name.get(category_to_coco[label], label)
    return label

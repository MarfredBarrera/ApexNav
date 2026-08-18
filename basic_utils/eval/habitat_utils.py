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

"""HM3D-OVON episodes (Yokoyama et al. 2024) for habitat 0.3.3's ObjectNav task.

OVON's files are ObjectNav-v1 with three differences that ObjectNavDatasetV1 cannot load:

- category maps are missing (top-level shell) or empty (per-scene files);
- episodes carry ``children_object_categories``, which ObjectGoalNavEpisode rejects;
- goal ``object_id`` is "<semantic name>_<semantic id>" (e.g. "painting_3") instead of the
  int semantic id. The suffix is the line id in ``<scene>.semantic.txt``, i.e. the
  instance id the semantic sensor reports, as HM3D-v2's ints are; the name goes to
  ``object_name`` (it differs from the category for ~13% of goals).

Importing this module registers dataset type ``OVON-v1`` (config/habitat_eval_ovon.yaml).
Category ids are filled from the episodes' categories (sorted) so ObjectGoalSensor has a
non-empty space; ApexNav itself uses the category name, not the id.
"""
import json

from habitat.core.registry import registry
from habitat.datasets.object_nav.object_nav_dataset import ObjectNavDatasetV1

CATEGORY_MAPS = ("category_to_task_category_id", "category_to_scene_annotation_category_id")


def fix_goal(goal):
    """OVON goal dict -> ObjectNav-v1 goal dict (int ``object_id``, name in ``object_name``)."""
    object_id = goal.get("object_id")
    if isinstance(object_id, str):
        name, _, number = object_id.rpartition("_")
        goal["object_id"] = int(number)
        goal.setdefault("object_name", name)
    return goal


def fix_json(json_str):
    """An OVON json string -> one ObjectNavDatasetV1.from_json accepts."""
    data = json.loads(json_str)
    for key in CATEGORY_MAPS:
        data.setdefault(key, {})
    for episode in data.get("episodes", []):
        episode.pop("children_object_categories", None)
        for goal in episode.get("goals") or []:
            fix_goal(goal)
    for goals in data.get("goals_by_category", {}).values():
        for goal in goals:
            fix_goal(goal)
    return json.dumps(data)


@registry.register_dataset(name="OVON-v1")
class OVONDatasetV1(ObjectNavDatasetV1):
    """ObjectNavDatasetV1 for HM3D-OVON files (see the module docstring)."""

    def __init__(self, config=None):
        self.category_to_task_category_id = {}
        self.category_to_scene_annotation_category_id = {}
        super().__init__(config)
        categories = sorted({episode.object_category for episode in self.episodes})
        self.category_to_task_category_id = {c: i for i, c in enumerate(categories)}
        self.category_to_scene_annotation_category_id = dict(self.category_to_task_category_id)

    def from_json(self, json_str, scenes_dir=None):
        super().from_json(fix_json(json_str), scenes_dir=scenes_dir)

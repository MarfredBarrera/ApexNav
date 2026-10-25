"""Draw a category-stratified random sample of episodes, as run_gate_e2e.sh indices.

    python scripts/sample_episodes.py --n 200 --seed 20261025 --out videos/sync_sample.json

Indices are 0-based positions in the evaluation's episode iterator (habitat_evaluation.py
``test_epi_num``), computed with the same config and iterator options, so they stay valid
for that config. Allocation is proportional to each target category's share (largest
remainder), then episodes are drawn without replacement within each category.
"""
import argparse
import json
import random

from hydra import compose, initialize
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import HabitatConfigPlugin, register_hydra_plugin
from habitat.datasets import make_dataset


def iterator_episodes(dataset_name):
    register_hydra_plugin(HabitatConfigPlugin)
    with initialize(version_base=None, config_path="../config"):
        cfg = patch_config(compose(config_name=f"habitat_eval_{dataset_name}"))
    hcfg = cfg.habitat
    dataset = make_dataset(id_dataset=hcfg.dataset.type, config=hcfg.dataset)
    options = {k.lower(): v for k, v in hcfg.environment.iterator_options.items()}
    options["seed"] = hcfg.seed
    iterator = dataset.get_episode_iterator(**options)
    return [next(iterator) for _ in range(len(dataset.episodes))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hm3dv2")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    episodes = iterator_episodes(args.dataset)
    by_category = {}
    for index, episode in enumerate(episodes):
        by_category.setdefault(episode.object_category, []).append(index)

    total = len(episodes)
    quota = {c: args.n * len(ix) / total for c, ix in by_category.items()}
    alloc = {c: int(q) for c, q in quota.items()}
    for c in sorted(quota, key=lambda c: (alloc[c] - quota[c], c))[: args.n - sum(alloc.values())]:
        alloc[c] += 1

    rng = random.Random(args.seed)
    chosen = []
    for category in sorted(by_category):
        chosen += rng.sample(by_category[category], alloc[category])
    rng.shuffle(chosen)  # run order: categories interleaved, so every block is mixed

    rows = [{"index": i, "scene_id": episodes[i].scene_id, "episode_id": episodes[i].episode_id,
             "category": episodes[i].object_category} for i in chosen]
    with open(args.out, "w") as f:
        json.dump({"dataset": args.dataset, "seed": args.seed, "n": args.n, "num_episodes": total,
                   "allocation": alloc, "population": {c: len(ix) for c, ix in by_category.items()},
                   "episodes": rows}, f, indent=1)
    print(f"{len(rows)} episodes from {total}: {alloc} -> {args.out}")


if __name__ == "__main__":
    main()

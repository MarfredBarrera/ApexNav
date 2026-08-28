"""
Running totals, metric tables and record files for an evaluation run.

record.txt holds the running *averages* after each episode; continue.txt holds
the running *totals* and is what a resumed run reads back to find its place.
Neither stores per-episode values - EpisodeRecorder's episodes.jsonl does that.
"""

import os
import time

from prettytable import PrettyTable

from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from params import RESULT_TYPES, result_dirname


class RunTotals:
    """Accumulates evaluation metrics across episodes and persists them."""

    def __init__(self, output_path, record_file_name, continue_file_name, flag_once):
        self.output_path = output_path
        self.record_path = os.path.join(output_path, record_file_name)
        self.continue_path = os.path.join(output_path, continue_file_name)

        (
            self.num_total,
            self.num_success,
            self.spl_all,
            self.soft_spl_all,
            self.distance_to_goal_all,
            self.distance_to_goal_reward_all,
            self.last_time,
        ) = read_record(self.continue_path, flag_once)

        self.start_time = time.time()
        self.result_counts = [0] * len(RESULT_TYPES)

    @property
    def elapsed(self):
        """Wall-clock seconds for this run, including any resumed time."""
        return time.time() - self.start_time + self.last_time

    def add(self, metrics, success):
        """Fold one finished episode into the totals."""
        if success == 1:
            self.num_success += 1
        self.num_total += 1
        self.spl_all += metrics["spl"]
        self.soft_spl_all += metrics["soft_spl"]
        self.distance_to_goal_all += metrics["distance_to_goal"]
        self.distance_to_goal_reward_all += metrics["distance_to_goal_reward"]

    def average_table(self):
        table = PrettyTable(["Metric", "Average"])
        n = max(self.num_total, 1)
        table.add_row(["Average Success", f"{self.num_success / n * 100:.2f}%"])
        table.add_row(["Average SPL", f"{self.spl_all / n * 100:.2f}%"])
        table.add_row(["Average Soft SPL", f"{self.soft_spl_all / n * 100:.2f}%"])
        table.add_row(
            ["Average Distance to Goal", f"{self.distance_to_goal_all / n:.4f}"]
        )
        return table

    def total_table(self):
        table = PrettyTable(["Metric", "Total"])
        table.add_row(["Total Success", f"{self.num_success}"])
        table.add_row(["Total SPL", f"{self.spl_all:.2f}"])
        table.add_row(["Total Soft SPL", f"{self.soft_spl_all:.2f}"])
        table.add_row(["Total Distance to Goal", f"{self.distance_to_goal_all:.4f}"])
        return table

    def report(self, result_text):
        """Print this episode's standing and return the two tables."""
        averages = self.average_table()
        print(averages)
        print(f"Episode {self.num_total} data written to {self.record_path}")
        print(f"Result: {result_text}")
        return averages, self.total_table()

    def persist(self, scene_id, episode_id, label, result_text, averages, totals):
        """Append this episode to record.txt and continue.txt."""
        for table, path in ((averages, self.record_path), (totals, self.continue_path)):
            write_record(
                scene_id,
                episode_id,
                table,
                result_text,
                label,
                self.num_total,
                self.elapsed,
                path,
            )

    def refresh_result_counts(self):
        """Count videos per outcome folder.

        Only meaningful when need_video is on, since the counts come from the
        mp4s sorted into per-result directories.
        """
        for i, folder in enumerate(RESULT_TYPES):
            self.result_counts[i] = count_files_in_directory(
                os.path.join(self.output_path, result_dirname(folder))
            )
        return self.result_counts

    def record_payload(self):
        """The /habitat/record array: aggregate metrics then per-result counts."""
        n = max(self.num_total, 1)
        payload = [
            self.num_success / n * 100,
            self.spl_all / n * 100,
            self.soft_spl_all / n * 100,
            self.distance_to_goal_all / n,
        ]
        payload.extend(self.refresh_result_counts())
        return payload

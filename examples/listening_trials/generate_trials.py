"""Generate answer keys and blank participant records for P1 listening trials."""

import csv
import random
from pathlib import Path


def generate(output_dir: Path, seed: int = 7, trials_per_task: int = 20) -> None:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_path = output_dir / "answer_key.csv"
    response_path = output_dir / "participant_responses.csv"
    rows = []
    for task, options in (("left_right", ("left", "right")),
                          ("sign", ("positive", "negative"))):
        for index in range(1, trials_per_task + 1):
            answer = rng.choice(options)
            rows.append({"trial_id": f"{task}-{index:02d}", "task": task,
                         "stimulus": answer, "correct_response": answer})
    with answer_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    fields = ["trial_id", "task", "headphones", "hrtf", "volume_db",
              "response", "response_time_s", "notes"]
    with response_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"trial_id": row["trial_id"], "task": row["task"],
                             "headphones": "", "hrtf": "", "volume_db": "",
                             "response": "", "response_time_s": "", "notes": ""})


if __name__ == "__main__":
    generate(Path(__file__).parent)

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publish_live_demo", ROOT / "scripts" / "publish_live_demo.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LiveDemoTests(unittest.TestCase):
    def test_incremental_publications_are_readable_and_physical_time_is_uniform(self):
        with tempfile.TemporaryDirectory() as directory:
            seen = []
            manifest = MODULE.publish_live(directory, frames=3, interval_s=1e-9,
                                           sleep_fn=lambda _: None,
                                           on_publish=lambda path: seen.append(json.loads(Path(path).read_text())))
            self.assertEqual(len(seen), 3)
            self.assertEqual([len(item["frames"]) for item in seen], [1, 2, 3])
            final = json.loads(Path(manifest).read_text())
            self.assertEqual(final["origin"], "synthetic")
            self.assertEqual(final["source_kind"], "synthetic")
            self.assertEqual(final["time_kind"], "dynamics")
            self.assertEqual(final["quantity"], "magnetization_direction")
            self.assertEqual(final["mask"], "all")
            times = [record["time_s"] for record in final["frames"]]
            self.assertEqual(times, [0.0, MODULE.STEP_S, 2 * MODULE.STEP_S])
            self.assertEqual([record["sequence"] for record in final["frames"]], [0, 1, 2])
            for record in final["frames"]:
                data = Path(record["file"]).read_bytes()
                self.assertEqual(record["sha256"], hashlib.sha256(data).hexdigest())

    def test_existing_nonempty_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "keep.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                MODULE.publish_live(directory, frames=2, interval_s=1e-9, sleep_fn=lambda _: None)

    def test_frames_and_interval_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                MODULE.publish_live(Path(directory) / "a", frames=1, interval_s=1)
            with self.assertRaises(ValueError):
                MODULE.publish_live(Path(directory) / "b", frames=2, interval_s=0)


if __name__ == "__main__":
    unittest.main()

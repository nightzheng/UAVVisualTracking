"""Workbench configuration, causality and task lifecycle contracts."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import time
import unittest
from unittest.mock import patch

from uav_visual_tracking.workbench.configuration import preset, preview, validate
from uav_visual_tracking.workbench.common import read, save
from uav_visual_tracking.workbench.pipeline import evaluate, source_path
from uav_visual_tracking.workbench.service import Runner
from uav_visual_tracking.workbench.timeline import eligible


class ConfigurationTests(unittest.TestCase):
    def test_speed_is_integrated_at_knots_and_reversal(self):
        cfg = preset("edge_sway")
        cfg["duration_s"] = 8
        cfg["rotor_motion"]["omega_points_deg_s"] = [[0, 1], [4, -1], [8, 0]]
        phase, speed = validate(cfg).sample([0, 2, 4, 6, 8])
        self.assertEqual(phase.tolist(), [0, 1, 0, -1.5, -2])
        self.assertEqual(speed.tolist(), [1, 0, -1, -.5, 0])

    def test_invalid_knots_speed_acceleration_and_initialization(self):
        for points in ([[0, 0], [0, 1]], [[0, 0], [72, 2]], [[0, 0], [1, 1], [72, 0]]):
            cfg = preset("edge_sway")
            cfg["rotor_motion"]["omega_points_deg_s"] = points
            with self.assertRaises(ValueError):
                validate(cfg)
        cfg = preset("edge_sway")
        cfg["rotor_motion"]["theta0_deg"] = None
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_recording_selection_rejects_unvalidated_paths(self):
        cfg = preset("edge_sway")
        for invalid in (dict(cfg, scene="../private"), dict(cfg, arm="/tmp"), []):
            with self.assertRaisesRegex(ValueError, "Invalid recording selection"):
                source_path(invalid, Path("result"))

    def test_frozen_and_seeded_curves(self):
        for scene in ("edge_sway", "slow_transition"):
            frozen = preset(scene)
            frozen["mode"] = "formal"
            preview(frozen)
            changed = preset(scene, True)
            self.assertEqual(changed, preset(scene, True))
            self.assertNotEqual(changed["rotor_motion"], frozen["rotor_motion"])
            changed["mode"] = "formal"
            with self.assertRaises(ValueError):
                validate(changed)


class DisplayTests(unittest.TestCase):
    def test_requires_application_and_both_clocks(self):
        plan = dict(station=0, source_capture_pts=1, plan_created_pts=1.1, valid_until=1.3,
                    plan_created_wall_s=2, valid_until_wall_s=2.3)
        ack = dict(command_sent_wall=2.1, applied_wall_s=2.15, applied_sim_s=1.2, publish_ok=True)
        self.assertTrue(eligible(plan, ack, 1.25, 2.2, 0))
        for t, wall, receipt in ((1.31, 2.2, ack), (1.25, 2.31, ack), (1.19, 2.2, ack),
                                 (1.25, 2.2, dict(ack, publish_ok=False))):
            self.assertFalse(eligible(plan, receipt, t, wall, 0))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name)
        self.runner = Runner(SimpleNamespace(repo=repo, results=repo / "results", ffmpeg=sys.executable, timeout=5))

    def finish(self, job, thread):
        thread.join(10)
        self.assertFalse(thread.is_alive())
        return read(job / "status.json")["state"]

    def test_validation_failure_keeps_snapshot(self):
        cfg = preset("edge_sway")
        cfg["duration_s"] = 0
        job, thread = self.runner.submit(cfg)
        self.assertEqual(self.finish(job, thread), "failed")
        self.assertEqual(read(job / "config.json"), cfg)
        self.assertFalse((job / "demo.mp4").exists())

    def test_nonmonotonic_clock_stops_before_evaluation(self):
        root = self.runner.root
        save(root / "frames.json", [{"pts": 0, "arrived_wall": 2}, {"pts": .05, "arrived_wall": 1}])
        save(root / "states.json", [{"sim_s": 0}, {"sim_s": .05}])
        with self.assertRaisesRegex(ValueError, "nonmonotonic.*arrived_wall"):
            evaluate(preset("edge_sway"), root, root)
        self.assertFalse((root / "result.json").exists())

    def test_failed_process_preserves_log(self):
        command = [sys.executable, "-c", "print('diagnostic'); raise SystemExit(7)"]
        with patch.object(self.runner, "command", return_value=(command, None)):
            job, thread = self.runner.submit(preset("edge_sway"))
            self.assertEqual(self.finish(job, thread), "failed")
        self.assertIn("diagnostic", (job / "task.log").read_text())
        self.assertFalse((job / "demo.mp4").exists())

    def test_cancel_and_single_active_job(self):
        command = [sys.executable, "-c", "import time; time.sleep(20)"]
        cfg = preset("edge_sway")
        with patch.object(self.runner, "command", return_value=(command, None)):
            job, thread = self.runner.submit(cfg)
            deadline = time.monotonic() + 3
            while not (job / "task.log").exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue((job / "task.log").is_file())
            with self.assertRaises(RuntimeError):
                self.runner.submit(cfg)
            self.runner.stop(job)
            self.assertEqual(self.finish(job, thread), "cancelled")
            self.assertTrue((job / "config.json").is_file())

    def test_write_failure_releases_lease(self):
        from uav_visual_tracking.workbench import service
        original = service.save

        def fail_snapshot(path, value):
            if path.name == "config.json":
                raise OSError("disk full")
            original(path, value)

        with patch.object(service, "save", side_effect=fail_snapshot):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.runner.submit(preset("edge_sway"))
        status = read(next(self.runner.root.glob("*/status.json")))
        self.assertEqual(status["state"], "failed")
        with self.runner.claim():
            pass


if __name__ == "__main__":
    unittest.main()

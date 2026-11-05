"""Validated configuration and exact rotor preview."""
import math
import random

import numpy as np

from .rotor_motion import RotorMotion

PRESETS = {
    "edge_sway": [[0, 1.15], [7, 1.15], [11, 0], [14, -1.35], [24, -1.35],
                  [30, 0], [35, 1.25], [45, 1.25], [55, 0], [63, -.7], [72, 0]],
    "slow_transition": [[0, .65], [15, .65], [23, 0], [28, 0], [38, -.65],
                        [52, -.25], [63, .4], [72, 0]],
}
ARMS = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}
MAX_OMEGA = 1.5
MAX_ACCELERATION = .5
MAX_PHASE = 15


def preset(scene, perturb=False):
    points = [row.copy() for row in PRESETS[scene]]
    if perturb:
        generator = random.Random(4)
        for row in points[1:-1]:
            row[1] += round(generator.uniform(-.025, .025), 4)
    return dict(version=1, mode="development", scene=scene, arm="D", duration_s=72,
                rotor_motion=dict(mode="piecewise_linear_omega", theta0_deg=0,
                                  omega_points_deg_s=points))


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(preset("edge_sway")):
        raise ValueError("Configuration fields do not match schema version 1")
    if (type(cfg["version"]) is not int or cfg["version"] != 1
            or cfg["mode"] not in ("development", "formal", "replay")
            or cfg["scene"] not in tuple(PRESETS) or cfg["arm"] not in tuple(ARMS)):
        raise ValueError("Invalid version, experiment mode, scene or strategy")
    duration = cfg["duration_s"]
    if not numeric(duration) or not 8 <= duration <= 72:
        raise ValueError("Duration must be between 8 and 72 seconds")
    rotor = cfg["rotor_motion"]
    if not isinstance(rotor, dict) or set(rotor) != {"mode", "theta0_deg", "omega_points_deg_s"}:
        raise ValueError("Invalid rotor configuration")
    if not numeric(rotor["theta0_deg"]):
        raise ValueError("Initial phase must be finite")
    if rotor["theta0_deg"] != 0:
        raise ValueError("The supplied C1 registration supports only initial phase 0 degrees")
    points = rotor["omega_points_deg_s"]
    if not isinstance(points, list) or not 2 <= len(points) <= 256 or any(
        not isinstance(row, list) or len(row) != 2 or not all(map(numeric, row)) for row in points
    ):
        raise ValueError("Provide 2–256 pairs of finite numeric keyframes")
    motion = RotorMotion(rotor)
    times, speeds = motion.points.T
    if times[-1] != duration:
        raise ValueError("Last keyframe must equal duration")
    if np.max(np.abs(speeds)) > MAX_OMEGA:
        raise ValueError(f"Speed exceeds ±{MAX_OMEGA} deg/s")
    slopes = np.diff(speeds) / np.diff(times)
    if np.max(np.abs(slopes)) > MAX_ACCELERATION:
        raise ValueError(f"Acceleration exceeds ±{MAX_ACCELERATION} deg/s²")
    crosses = np.flatnonzero(speeds[:-1] * speeds[1:] < 0)
    extrema = np.r_[times, times[crosses] - speeds[crosses] / slopes[crosses]]
    if np.max(np.abs(motion.sample(extrema)[0])) > MAX_PHASE:
        raise ValueError(f"Integrated phase exceeds ±{MAX_PHASE} degrees")
    if cfg["mode"] in ("formal", "replay") and (points != PRESETS[cfg["scene"]] or duration != 72):
        raise ValueError("Formal and replay configurations require the original frozen curve and duration")
    return motion


def preview(cfg):
    motion = validate(cfg)
    times = np.unique(np.r_[np.linspace(0, cfg["duration_s"], 361), motion.points[:, 0]])
    phase, omega = motion.sample(times)
    return dict(time_s=times.tolist(), phase_deg=phase.tolist(), omega_deg_s=omega.tolist(),
                acceleration_deg_s2=(np.diff(motion.points[:, 1]) / np.diff(motion.points[:, 0])).tolist())

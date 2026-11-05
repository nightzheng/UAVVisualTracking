"""Execution stages shared by the CLI and local service."""
import argparse
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from .common import ARCH, read, save, sha
from .configuration import ARMS, PRESETS, validate

G = ARCH / "stage_u3g_joint_uav_gimbal_01"


def source_path(cfg, job):
    if (not isinstance(cfg, dict) or cfg.get("scene") not in tuple(PRESETS)
            or cfg.get("arm") not in tuple(ARMS)
            or cfg.get("mode") not in ("development", "formal", "replay")):
        raise ValueError("Invalid recording selection")
    root = G / "formal" if cfg["mode"] == "replay" else job / "recording"
    return root / cfg["scene"] / cfg["arm"]


def prepare(cfg, job, plugin):
    template = G / "formal" / cfg["scene"] / cfg["arm"]
    for name in ("consumer_spec.json", "local.sdf", "config_snapshot.yaml"):
        if not (template / name).is_file():
            raise ValueError(f"Missing frozen C1 input: {template / name}")
    locked = read(G / "formal" / "LOCK.json")["inputs"][f'{cfg["scene"]}/{cfg["arm"]}']
    for name, digest in locked.items():
        if sha(template / name) != digest:
            raise ValueError(f"Frozen template checksum mismatch: {name}")
    if not plugin.is_file():
        raise ValueError(f"Missing Gazebo motion plugin: {plugin}")
    run = source_path(cfg, job)
    run.mkdir(parents=True)
    spec = read(template / "consumer_spec.json")
    if not spec.get("G6_C1_admitted") or not spec.get("banks") or spec.get("initial_phase_deg") != 0:
        raise ValueError("Missing admitted C1 registration or initial phase")
    if {"rotor_motion", "omega_points_deg_s", "evaluation"} & spec.keys():
        raise ValueError("Future motion and independent evaluation cannot enter the live specification")
    spec["source"]["duration_s"] = cfg["duration_s"]
    save(run / "consumer_spec.json", spec)

    world = ET.parse(template / "local.sdf")
    for node in world.iter():
        if node.tag in ("uri", "albedo_map") and node.text and "/.local/research_archive/" in node.text:
            asset = (ARCH / node.text.split("/.local/research_archive/", 1)[1]).resolve()
            if not asset.is_relative_to(ARCH) or not asset.is_file():
                raise ValueError(f"Missing world asset: {asset}")
            node.text = str(asset)
    motion_plugin = world.find(".//plugin[@name='U3Motion']")
    motion_plugin.set("filename", str(plugin))
    points = cfg["rotor_motion"]["omega_points_deg_s"]
    motion_plugin.find("omega_points").text = " ".join(format(value, ".17g") for row in points for value in row)
    world.write(run / "local.sdf", encoding="utf-8")
    snapshot = yaml.safe_load((template / "config_snapshot.yaml").read_text(encoding="utf-8"))
    snapshot["source"]["duration_s"] = cfg["duration_s"]
    snapshot["rotor_motion"] = cfg["rotor_motion"]
    (run / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot), encoding="utf-8")
    save(run / "capture_plan.json", dict(duration_s=cfg["duration_s"], fps=20,
         effective_omega_points_deg_s=points, theta0_deg=0, world_sha256=sha(run / "local.sdf")))
    save(job / "input_hashes.json", {name: sha(run / name) for name in
         ("local.sdf", "config_snapshot.yaml", "consumer_spec.json", "capture_plan.json")})
    return run


def evaluate(cfg, job, run):
    from .score_f import split_evaluate
    from .score_g import audit

    frames = read(run / "frames.json")
    states = read(run / "states.json")
    for rows, key in ((frames, "pts"), (frames, "arrived_wall"), (states, "sim_s")):
        if len(rows) < 2 or any(b[key] <= a[key] for a, b in zip(rows, rows[1:])):
            raise ValueError(f"Missing or nonmonotonic recording clock: {key}")
    if any(round(frame["pts"] * 20) != i for i, frame in enumerate(frames)):
        raise ValueError("Discontinuous 20 Hz recording")
    if cfg["mode"] != "replay":
        motion = validate(cfg)
        times = [state["sim_s"] for state in states]
        phase, speed = motion.sample(times)
        actual_phase = np.rad2deg([state["joint_q_rad"] for state in states])
        actual_speed = np.rad2deg([state["joint_v_rad_s"] for state in states])
        phase_error = float(np.max(np.abs(phase - actual_phase)))
        speed_error = float(np.max(np.abs(speed - actual_speed)))
        if phase_error > .02 or speed_error > .02:
            raise ValueError("Measured Gazebo motion does not match the requested curve")
        indices = sorted({0, len(frames) // 2, len(frames) - 1})
        reference = G / "formal" / cfg["scene"] / cfg["arm"]
        rgb = []
        for i in indices:
            file = frames[i]["file"]
            previous = reference / file
            rgb.append(dict(frame=i, simulation_s=frames[i]["pts"], file=file,
                            sha256=sha(run / file),
                            frozen_sha256=sha(previous) if previous.is_file() else None))
        save(job / "motion_verification.json", dict(
            maximum_phase_error_deg=phase_error, maximum_speed_error_deg_s=speed_error,
            actual_rgb_samples=rgb))
    evaluation = split_evaluate(run, job)
    joint = audit(run, job)
    if cfg["mode"] == "replay":
        frozen = read(run / "IMAGING_EVALUATION.json")
        for key in ("region_count", "quality_count", "coverage", "joint_bad", "checks"):
            if evaluation[key] != frozen[key]:
                raise ValueError(f"Frozen replay metric differs: {key}")
    photos = {row["station"]: row for row in evaluation["photos"]}
    transitions = {row["station"] + 1: row["result"] for row in read(run / "transitions.json")}
    spec = read(run / "consumer_spec.json")
    regions = []
    for i, bank in enumerate(spec["banks"], 1):
        photo = photos.get(i)
        completed = bool(photo and photo["region_complete"])
        reason = None if completed else ("independent_coverage_or_quality_rejected" if photo else
                                         transitions.get(i, "experiment_ended_before_capture"))
        regions.append(dict(station=i, region_eta=bank["region_eta"], complete=completed,
                            reason=reason, photo=photo))
    save(job / "result.json", dict(source=cfg["mode"], evaluation=evaluation, joint=joint, regions=regions))
    with (job / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["simulation_s", "camera_x_m", "camera_y_m", "camera_z_m",
                         "camera_qx", "camera_qy", "camera_qz", "camera_qw", "rotor_rad_s"])
        writer.writerows([row["sim_s"], *row["camera_pose"], row["joint_v_rad_s"]] for row in states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("check", "prepare", "run", "evaluate", "render"))
    parser.add_argument("job", type=Path)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    cfg = read(args.job / "config.json")
    validate(cfg)
    run = source_path(cfg, args.job)
    if args.stage == "check":
        import cv2
        if not shutil.which("gz"):
            raise RuntimeError("Gazebo executable unavailable")
        import gz.transport13
        import gz.msgs10.image_pb2
        if not args.plugin.is_file():
            raise ValueError(f"Missing Gazebo motion plugin: {args.plugin}")
        subprocess.run(["gz", "sim", "--versions"], check=True, timeout=15)
        print("Gazebo executable, Python transport and motion plugin found")
    elif args.stage == "prepare":
        prepare(cfg, args.job, args.plugin)
    elif args.stage == "run":
        from .live_g import main as live
        sys.argv[1:] = [str(run)]
        live()
    elif args.stage == "evaluate":
        evaluate(cfg, args.job, run)
    else:
        from .render_h import render

        output = args.job / "encoding.mp4"
        render(run, output, args.ffmpeg, args.job)
        output.with_suffix(".frames.json").rename(args.job / "demo.frames.json")
        output.rename(args.job / "demo.mp4")


if __name__ == "__main__":
    main()

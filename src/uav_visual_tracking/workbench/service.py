"""Loopback experiment service and serial CLI runner."""
import argparse
from importlib.metadata import version
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from threading import Event, Lock, Thread
import time
from uuid import uuid4
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from .common import ARCH, read, save
from .configuration import PRESETS, preset, preview, validate
from .pipeline import source_path

MODULE = "uav_visual_tracking.workbench.pipeline"
TERMINAL = {"completed", "failed", "cancelled"}
ARTIFACTS = {".json", ".csv", ".png", ".mp4", ".log", ".npz", ".yaml"}


class Cancelled(Exception):
    pass


class Runner:
    def __init__(self, options):
        self.options = options
        self.root = options.results.resolve()
        if self.root.is_relative_to(ARCH):
            raise ValueError("Results must be written outside the frozen archive")
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.cancel = Event()
        self.active = None
        self.volatile_status = {}

    def status(self, job, state, error=None):
        path = job / "status.json"
        data = dict(id=job.name, state=state, error=error)
        self.volatile_status[job.name] = data
        temporary = job / "status.tmp"
        save(temporary, data)
        for attempt in range(5):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(.05)
        print(job.name, state, error or "", flush=True)

    def claim(self):
        lock_path = self.options.repo / ".local" / "workbench.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lock_path.open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError("Another experiment is running in this repository") from exc
        return stream

    def submit(self, cfg):
        with self.lock:
            if self.active:
                raise RuntimeError("Another experiment is running")
            lease = self.claim()
            try:
                job = self.root / uuid4().hex
                job.mkdir()
                self.status(job, "created")
                save(job / "config.json", cfg)
                self.active = job
                self.cancel.clear()
                thread = Thread(target=self.execute, args=(job, cfg, lease))
                thread.start()
                return job, thread
            except BaseException:
                self.active = None
                lease.close()
                if "job" in locals() and job.is_dir():
                    try:
                        self.status(job, "failed", "Unable to persist the experiment configuration")
                    except OSError:
                        pass
                raise

    def stop(self, job):
        with self.lock:
            if job != self.active:
                raise ValueError("Experiment is no longer active")
            self.cancel.set()

    def command(self, job, stage, live=False):
        args = ["-B", "-m", MODULE, stage, str(job)]
        if stage == "render":
            args.extend(["--ffmpeg", self.options.ffmpeg])
        env = os.environ.copy()
        source = str(self.options.repo / "src")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (source, env.get("PYTHONPATH"))))
        env["UAV_WORKBENCH_ARCHIVE"] = str(ARCH)
        if live:
            plugin = self.options.plugin
            if sys.platform == "win32":
                wsl = [self.options.wsl, "-d", self.options.distro, "--exec"]
                def translate(path):
                    return subprocess.check_output([*wsl, "wslpath", "-a", "-u", str(path)],
                                                   timeout=20).decode("utf-8").strip()

                args[4] = translate(job)
                script = ('source /opt/ros/jazzy/setup.bash && export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}" '
                          'UAV_WORKBENCH_ARCHIVE="$2" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
                          '&& shift 2 && exec "$@"')
                return [*wsl, "bash", "-c", script, "workbench", translate(self.options.repo / "src"),
                        translate(ARCH), self.options.linux_python, *args, "--plugin", translate(plugin)], env
            return [self.options.linux_python, *args, "--plugin", str(plugin)], env
        return [sys.executable, *args], env

    def stage(self, job, stage, live=False):
        if self.cancel.is_set():
            raise Cancelled()
        cmd, env = self.command(job, stage, live)
        with (job / "task.log").open("ab", buffering=0) as log:
            log.write(f"\n--- {stage} ---\n".encode())
            with subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT) as proc:
                deadline = time.monotonic() + self.options.timeout
                while proc.poll() is None:
                    cancelled = self.cancel.wait(.2)
                    if cancelled or time.monotonic() >= deadline:
                        proc.stdin.close()
                        if stage == "run":
                            try:
                                proc.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                pass
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                        if cancelled:
                            raise Cancelled()
                        raise TimeoutError(f"Stage timed out: {stage}")
                if self.cancel.is_set():
                    raise Cancelled()
                if proc.returncode:
                    raise RuntimeError(f"{stage} exited with status {proc.returncode}; see task.log")

    def execute(self, job, cfg, lease):
        try:
            self.status(job, "validating")
            validate(cfg)
            if not shutil.which(self.options.ffmpeg):
                raise ValueError("FFmpeg unavailable")
            self.status(job, "preparing")
            if cfg["mode"] != "replay":
                self.stage(job, "check", live=True)
                self.stage(job, "prepare", live=True)
                self.status(job, "running")
                self.stage(job, "run", live=True)
            self.status(job, "evaluating")
            self.stage(job, "evaluate")
            self.status(job, "rendering")
            self.stage(job, "render")
            self.status(job, "completed")
        except Cancelled:
            self.status(job, "cancelled")
        except Exception as exc:
            try:
                self.status(job, "failed", str(exc))
            except OSError as status_error:
                print(f"Failed to persist status for {job.name}: {status_error}; {exc}", file=sys.stderr)
        finally:
            with self.lock:
                lease.close()
                self.active = None


def handler_for(runner, web):
    class Handler(BaseHTTPRequestHandler):
        def json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def file(self, file):
            if not file.is_file():
                self.send_error(404)
                return
            with file.open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
                start, end = 0, size - 1
                header = self.headers.get("Range")
                if header:
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", header)
                    if not match:
                        self.send_error(416)
                        return
                    start = int(match[1])
                    end = min(end, int(match[2])) if match[2] else end
                    if start > end:
                        self.send_error(416)
                        return
                self.send_response(206 if header else 200)
                self.send_header("Content-Type", mimetypes.guess_type(file)[0] or "application/octet-stream")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Accept-Ranges", "bytes")
                if header:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                remaining = end - start + 1
                self.send_header("Content-Length", str(remaining))
                self.end_headers()
                stream.seek(start)
                while remaining > 0:
                    block = stream.read(min(remaining, 65536))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def local_request(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            origin = self.headers.get("Origin")
            if host in allowed and (origin is None or origin == f"http://{host}"):
                return True
            self.send_error(403)
            return False

        def do_GET(self):
            if not self.local_request():
                return
            try:
                path = unquote(urlsplit(self.path).path)
                if path == "/api/version":
                    return self.json({"version": version("uav-visual-tracking")})
                if path == "/api/presets":
                    return self.json({scene: [preset(scene), preset(scene, True)] for scene in PRESETS})
                if path == "/api/jobs":
                    jobs = sorted(runner.root.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    rows = {job.parent.name: read(job) for job in jobs
                            if re.fullmatch("[0-9a-f]{32}", job.parent.name)}
                    rows.update(runner.volatile_status)
                    return self.json(list(rows.values()))
                match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})(?:/(.+))?", path)
                if match:
                    job = runner.root / match[1]
                    tail = match[2]
                    if tail is None:
                        return self.json(runner.volatile_status.get(job.name) or read(job / "status.json"))
                    if tail == "logs":
                        cfg = read(job / "config.json")
                        logs = {}
                        files = [job / "task.log"]
                        try:
                            files.append(source_path(cfg, job) / "server.log")
                        except ValueError:
                            pass
                        for file in files:
                            if file.is_file():
                                with file.open("rb") as stream:
                                    stream.seek(max(0, file.stat().st_size - 16000))
                                    logs[file.name] = stream.read().decode("utf-8", errors="replace")
                        return self.json(logs)
                    base = job
                    if tail.startswith("recording/"):
                        base = source_path(read(job / "config.json"), job)
                        tail = tail.removeprefix("recording/")
                    file = (base / tail).resolve()
                    if tail == "encoding.mp4":
                        return self.send_error(404)
                    if tail == "demo.mp4":
                        state = runner.volatile_status.get(job.name) or read(job / "status.json")
                        if state["state"] != "completed":
                            return self.send_error(404)
                    if file.is_relative_to(base.resolve()) and file.suffix.lower() in ARTIFACTS:
                        return self.file(file)
                    return self.send_error(404)
                file = (web / path.lstrip("/")).resolve() if path != "/" else web / "index.html"
                if file.is_relative_to(web):
                    return self.file(file)
                self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                return
            except (OSError, ValueError) as exc:
                self.json({"error": str(exc)}, 404)

        def do_POST(self):
            if not self.local_request():
                return
            try:
                if self.headers.get_content_type() != "application/json":
                    return self.json({"error": "Expected application/json"}, 415)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    return self.json({"error": "Invalid request size"}, 400)
                payload = json.loads(self.rfile.read(length))
                if self.path == "/api/preview":
                    return self.json(preview(payload))
                if self.path == "/api/jobs":
                    job, _ = runner.submit(payload)
                    return self.json({"id": job.name}, 202)
                match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/cancel", self.path)
                if match:
                    runner.stop(runner.root / match[1])
                    return self.json({"id": match[1]}, 202)
                self.send_error(404)
            except RuntimeError as exc:
                self.json({"error": str(exc)}, 409)
            except (ValueError, TypeError, OSError) as exc:
                self.json({"error": str(exc)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(prog="uav-workbench")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {version('uav-visual-tracking')}")
    parser.add_argument("command", choices=("serve", "run"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results", type=Path, default=Path(".local/u4-results"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--plugin", type=Path, default=ARCH / "stage_u3d_visual_inspection_01/build/libU3Motion.so")
    parser.add_argument("--distro", default="Ubuntu-24.04")
    parser.add_argument("--wsl", default="wsl.exe")
    parser.add_argument("--linux-python", default="python3")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--port", type=int, default=8765)
    options = parser.parse_args()
    options.repo = options.repo.resolve()
    options.plugin = options.plugin.resolve()
    runner = Runner(options)
    if options.command == "run":
        if options.config is None:
            parser.error("--config is required")
        job, thread = runner.submit(read(options.config))
        try:
            thread.join()
        except KeyboardInterrupt:
            runner.cancel.set()
            thread.join()
        return 0 if read(job / "status.json")["state"] == "completed" else 1
    web = options.repo / "workbench-ui/dist"
    if not (web / "index.html").is_file():
        parser.error("Build the Vue frontend in workbench-ui first")
    with ThreadingHTTPServer(("127.0.0.1", options.port), handler_for(runner, web)) as server:
        print(f"http://127.0.0.1:{options.port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            runner.cancel.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())

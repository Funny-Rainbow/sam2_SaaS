import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass


READY_SUBSTRINGS = (
    "Application startup complete.",
    "Uvicorn running on",
)


def _now() -> float:
    return time.monotonic()


def _open_listen_socket(host: str, port: int, backlog: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    sock.set_inheritable(True)
    return sock


def _tee_lines(pipe, out_stream, ready_event: threading.Event):
    try:
        for line in iter(pipe.readline, ""):
            out_stream.write(line)
            out_stream.flush()
            if not ready_event.is_set() and any(s in line for s in READY_SUBSTRINGS):
                ready_event.set()
    finally:
        try:
            pipe.close()
        except Exception:
            pass


@dataclass
class Child:
    proc: subprocess.Popen
    ready_event: threading.Event
    _threads: tuple[threading.Thread, threading.Thread]


def _start_child(
    *,
    python_exe: str,
    app: str,
    fd: int,
    app_dir: str,
    log_level: str,
    workers: int,
    timeout_graceful_shutdown: int,
    limit_concurrency: int | None,
    limit_max_requests: int | None,
) -> Child:
    cmd = [
        python_exe,
        "-m",
        "uvicorn",
        app,
        "--app-dir",
        app_dir,
        "--fd",
        str(fd),
        "--log-level",
        log_level,
        "--workers",
        str(workers),
        "--timeout-graceful-shutdown",
        str(timeout_graceful_shutdown),
    ]
    if limit_concurrency is not None:
        cmd += ["--limit-concurrency", str(limit_concurrency)]
    if limit_max_requests is not None:
        cmd += ["--limit-max-requests", str(limit_max_requests)]

    ready_event = threading.Event()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        pass_fds=(fd,),
        close_fds=True,
        cwd=app_dir,
        env=os.environ.copy(),
    )

    assert proc.stdout is not None
    assert proc.stderr is not None
    t_out = threading.Thread(target=_tee_lines, args=(proc.stdout, sys.stdout, ready_event), daemon=True)
    t_err = threading.Thread(target=_tee_lines, args=(proc.stderr, sys.stderr, ready_event), daemon=True)
    t_out.start()
    t_err.start()
    return Child(proc=proc, ready_event=ready_event, _threads=(t_out, t_err))


def _stop_child(proc: subprocess.Popen, *, graceful_seconds: int, kill_seconds: int) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=graceful_seconds)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        proc.send_signal(signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=kill_seconds)
    except subprocess.TimeoutExpired:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SAM2 API with periodic restart (12h) without dropping connects.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to launch uvicorn with.")
    parser.add_argument("--app", default="sam2_segment_api:app", help="ASGI app import string for uvicorn.")
    parser.add_argument(
        "--app-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory added to PYTHONPATH for importing the ASGI app.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the listening socket in the supervisor.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the listening socket in the supervisor.")
    parser.add_argument("--backlog", type=int, default=4096, help="TCP listen backlog.")

    parser.add_argument("--restart-interval-seconds", type=int, default=12 * 60 * 60, help="Time-based restart interval.")
    parser.add_argument("--restart-ready-timeout-seconds", type=int, default=180, help="Max wait for new child readiness.")
    parser.add_argument("--restart-retry-seconds", type=int, default=60, help="Retry interval if rolling restart fails.")

    parser.add_argument("--log-level", default="info", help="Uvicorn log level.")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers. Keep 1 for single-GPU workloads.")
    parser.add_argument(
        "--timeout-graceful-shutdown",
        type=int,
        default=3600,
        help="Uvicorn graceful shutdown timeout (seconds).",
    )
    parser.add_argument(
        "--old-process-graceful-seconds",
        type=int,
        default=3600,
        help="Max seconds to wait for old process after SIGTERM before SIGKILL.",
    )
    parser.add_argument("--old-process-kill-seconds", type=int, default=10, help="Max seconds to wait after SIGKILL.")

    parser.add_argument("--limit-concurrency", type=int, default=None, help="Uvicorn limit concurrency (optional).")
    parser.add_argument("--limit-max-requests", type=int, default=None, help="Uvicorn limit max requests (optional).")

    args = parser.parse_args()

    sock = _open_listen_socket(args.host, args.port, args.backlog)
    fd = sock.fileno()

    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    child = _start_child(
        python_exe=args.python,
        app=args.app,
        fd=fd,
        app_dir=args.app_dir,
        log_level=args.log_level,
        workers=args.workers,
        timeout_graceful_shutdown=args.timeout_graceful_shutdown,
        limit_concurrency=args.limit_concurrency,
        limit_max_requests=args.limit_max_requests,
    )
    if not child.ready_event.wait(timeout=args.restart_ready_timeout_seconds):
        if child.proc.poll() is None:
            _stop_child(
                child.proc,
                graceful_seconds=min(args.old_process_graceful_seconds, 30),
                kill_seconds=args.old_process_kill_seconds,
            )
        raise RuntimeError("Initial uvicorn start did not become ready in time.")

    next_restart_at = _now() + args.restart_interval_seconds

    while not stop_event.is_set():
        if child.proc.poll() is not None:
            # Crash/restart: bring it back as soon as possible.
            child = _start_child(
                python_exe=args.python,
                app=args.app,
                fd=fd,
                app_dir=args.app_dir,
                log_level=args.log_level,
                workers=args.workers,
                timeout_graceful_shutdown=args.timeout_graceful_shutdown,
                limit_concurrency=args.limit_concurrency,
                limit_max_requests=args.limit_max_requests,
            )
            child.ready_event.wait(timeout=args.restart_ready_timeout_seconds)
            next_restart_at = _now() + args.restart_interval_seconds
            time.sleep(1)
            continue

        if _now() >= next_restart_at:
            replacement = _start_child(
                python_exe=args.python,
                app=args.app,
                fd=fd,
                app_dir=args.app_dir,
                log_level=args.log_level,
                workers=args.workers,
                timeout_graceful_shutdown=args.timeout_graceful_shutdown,
                limit_concurrency=args.limit_concurrency,
                limit_max_requests=args.limit_max_requests,
            )

            ready = replacement.ready_event.wait(timeout=args.restart_ready_timeout_seconds)
            if replacement.proc.poll() is not None:
                ready = False

            if ready:
                _stop_child(
                    child.proc,
                    graceful_seconds=args.old_process_graceful_seconds,
                    kill_seconds=args.old_process_kill_seconds,
                )
                child = replacement
                next_restart_at = _now() + args.restart_interval_seconds
            else:
                _stop_child(
                    replacement.proc,
                    graceful_seconds=min(args.old_process_graceful_seconds, 30),
                    kill_seconds=args.old_process_kill_seconds,
                )
                next_restart_at = _now() + args.restart_retry_seconds

        time.sleep(1)

    _stop_child(
        child.proc,
        graceful_seconds=args.old_process_graceful_seconds,
        kill_seconds=args.old_process_kill_seconds,
    )
    try:
        sock.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

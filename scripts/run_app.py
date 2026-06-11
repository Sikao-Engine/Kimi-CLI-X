#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_app.py — Launch both the Kimix backend and the TypeScript/Vite frontend as subprocesses.

Backend:  FastAPI server via `kimix serve` (default http://127.0.0.1:4096)
         Uses DummySessionManager by default; pass --be-real for live SDK sessions.
Frontend: Vite dev server via `npm run dev` (default http://localhost:5173)
         Connects to the backend and provides a web UI mirroring sse_cli.py logic.

Usage:
    uv run scripts/run_app.py [--host HOST] [--port PORT] [--fe-port PORT] [--be-real] [--build]

Ctrl+C gracefully terminates both servers.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

# ── Defaults ──────────────────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4096
DEFAULT_FE_PORT = 5173  # matches vite.config.ts server.port

# Paths relative to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(REPO_ROOT, "src", "app")


def _resolve_uv() -> str:
    """Return the `uv` executable path."""
    return "uv"


def _resolve_npm() -> str:
    """Return `npm` (or `npm.cmd` on Windows)."""
    if sys.platform == "win32":
        return "npm.cmd"
    return "npm"


def _check_prerequisites() -> list[str]:
    """Check that required tools and dependencies are available.

    Returns a list of warning strings (empty = all good).
    """
    warnings: list[str] = []

    if not os.path.isdir(os.path.join(APP_DIR, "node_modules")):
        warnings.append(
            "node_modules not found in src/app. "
            "Run `npm install` in src/app before starting the frontend."
        )

    # Check npm/node are on PATH
    npm = _resolve_npm()
    try:
        subprocess.run(
            [npm, "--version"],
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        warnings.append(
            f"`{npm}` not found on PATH. Install Node.js and npm to run the frontend."
        )

    return warnings


def _build_frontend(npm: str) -> bool:
    """Run `npm run build` in the frontend directory. Returns True on success."""
    print("[run_app] Building frontend...")
    result = subprocess.run(
        [npm, "run", "build"],
        cwd=APP_DIR,
        capture_output=False,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Kimix backend + TypeScript/Vite frontend.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Backend & frontend bind host (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Backend bind port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--fe-port",
        type=int,
        default=DEFAULT_FE_PORT,
        help=f"Frontend dev-server port (default: {DEFAULT_FE_PORT})",
    )
    parser.add_argument(
        "--be-real",
        action="store_true",
        help=(
            "Use the real SessionManager (live SDK sessions) instead of the "
            "default DummySessionManager. The real backend requires API credentials."
        ),
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run `npm run build` before starting the dev server.",
    )
    args = parser.parse_args()

    # ── Prerequisite checks ───────────────────────────────────────
    warnings = _check_prerequisites()
    for w in warnings:
        print(f"[run_app] WARNING: {w}", file=sys.stderr)

    uv = _resolve_uv()
    npm = _resolve_npm()

    # ── Optional: build frontend first ────────────────────────────
    if args.build:
        if not _build_frontend(npm):
            print("[run_app] ERROR: Frontend build failed.", file=sys.stderr)
            sys.exit(1)

    # ── Backend ───────────────────────────────────────────────────
    be_cmd = [
        uv, "run", "kimix", "serve",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.be_real:
        # Use the real app (SessionManager with live SDK) instead of DummySessionManager.
        # We inject the real app module via a small inline script.
        be_cmd = [
            uv, "run", "python", "-c",
            (
                "import uvicorn; "
                "from kimix.server.app import create_app; "
                "app = create_app(); "
                f"uvicorn.run(app, host='{args.host}', port={args.port}, log_level='info')"
            ),
        ]
        print("[run_app] Starting backend (REAL SessionManager) ...")
    else:
        print(f"[run_app] Starting backend (dummy): {' '.join(be_cmd)}")

    be_proc = subprocess.Popen(
        be_cmd,
        cwd=REPO_ROOT,
    )

    # ── Frontend ──────────────────────────────────────────────────
    # Vite CLI: npm run dev -- --host <host> --port <port>
    fe_cmd = [
        npm, "run", "dev",
        "--", "--host", args.host, "--port", str(args.fe_port),
    ]
    print(f"[run_app] Starting frontend: {' '.join(fe_cmd)}")
    fe_proc = subprocess.Popen(
        fe_cmd,
        cwd=APP_DIR,
    )

    # ── Print banner ──────────────────────────────────────────────
    time.sleep(2.0)  # give both servers a moment to start
    print()
    print("=" * 64)
    print("  Kimix App — Backend + Frontend")
    print("=" * 64)
    print(f"  Backend      : http://{args.host}:{args.port}")
    print(f"  API Docs     : http://{args.host}:{args.port}/docs")
    print(f"  Frontend     : http://{args.host}:{args.fe_port}")
    if args.be_real:
        print(f"  Backend mode : REAL (live SDK sessions)")
    else:
        print(f"  Backend mode : DUMMY (stub session manager)")
    print("=" * 64)
    print("  Commands (frontend): /new /abort /status /sessions /messages")
    print("                       /clear /compact /export /exit")
    print("  Press Ctrl+C to stop both servers.")
    print()

    # ── Shutdown handler ──────────────────────────────────────────
    procs: list[subprocess.Popen] = [be_proc, fe_proc]

    def _shutdown(signum: int, frame: object) -> None:  # noqa: ARG001
        print("\n[run_app] Shutting down...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        # Give processes a moment to exit gracefully
        time.sleep(2)
        for p in procs:
            if p.poll() is None:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Wait for both processes ───────────────────────────────────
    try:
        be_proc.wait()
        fe_proc.wait()
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)


if __name__ == "__main__":
    main()

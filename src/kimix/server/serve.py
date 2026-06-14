# -*- coding: utf-8 -*-
"""CLI entry point for `kimix serve` — starts the opencode-style HTTP server."""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4096


def serve_cli(args: argparse.Namespace) -> None:
    """Start the kimix HTTP server with uvicorn."""
    host = getattr(args, "host", DEFAULT_HOST)
    port = getattr(args, "port", DEFAULT_PORT)

    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is required for `kimix serve`. "
            "Install with: pip install uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    from kimix.server.app import create_app

    app = create_app()

    print(f"kimix server listening on http://{host}:{port}")
    print(f"API docs (Swagger UI): http://{host}:{port}/docs")
    print(f"OpenAPI schema: http://{host}:{port}/openapi.json")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_graceful_shutdown=5,
    )

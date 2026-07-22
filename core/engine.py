"""
engine.py — Headless Simulation Entry Point
============================================

Parses CLI arguments and launches the WebSocket server which runs the
simulation in a headless loop, broadcasting snapshots to connected web clients.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D Ecosystem Simulation Engine")
    parser.add_argument(
        "-p", "--path", "--load",
        dest="load_path",
        type=str,
        default=None,
        help="Path to saved genomes JSON file to start the simulation from.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for the WebSocket/HTTP server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the WebSocket/HTTP server.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the headless simulation server."""
    args = parse_args()

    from server import run_server
    run_server(
        host=args.host,
        port=args.port,
        load_path=args.load_path,
    )


if __name__ == "__main__":
    main()

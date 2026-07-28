"""
server.py — WebSocket + HTTP Server for Headless Ecosystem Simulation
=====================================================================

Runs the simulation core in an asyncio event loop, serves the web interface,
and broadcasts state snapshots over WebSocket to connected frontend clients.
Handles bidirectional messaging for hot-swapping constants and control commands.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import os
import socketserver
import threading
import websockets
from typing import Any

from core.serialization import build_full_snapshot, build_aggregate_snapshot, encode
from core.simulation import Simulation

logging.basicConfig(level=logging.INFO, format="[SERVER] %(levelname)s: %(message)s")
logger = logging.getLogger("server")

CLIENTS: set[websockets.WebSocketServerProtocol] = set()
SIMULATION: Simulation | None = None
IS_PAUSED: bool = False


class StaticFileHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static files from the project directory (web/ and assets/)."""

    def translate_path(self, path: str) -> str:
        # Serve root request from web/index.html
        if path == "/" or path == "/index.html":
            return os.path.join(os.path.dirname(__file__), "web", "index.html")
        
        # Check assets or web folder
        clean_path = path.lstrip("/")
        web_path = os.path.join(os.path.dirname(__file__), "web", clean_path)
        if os.path.exists(web_path):
            return web_path
        
        asset_path = os.path.join(os.path.dirname(__file__), clean_path)
        if os.path.exists(asset_path):
            return asset_path
            
        return super().translate_path(path)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy HTTP GET logs
        pass


def start_http_server(host: str, port: int) -> int:
    """Start static HTTP file server in a background thread."""
    http_port = port + 1
    
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = ReusableTCPServer((host, http_port), StaticFileHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Static HTTP Server running at http://{host}:{http_port}/")
    return http_port


async def handle_client_message(websocket: websockets.WebSocketServerProtocol, raw_msg: str) -> None:
    global IS_PAUSED, SIMULATION
    if SIMULATION is None:
        return

    try:
        data = json.loads(raw_msg)
    except json.JSONDecodeError:
        return

    msg_type = data.get("type")

    if msg_type == "set_speed":
        direction = data.get("direction")
        if direction == "up":
            SIMULATION.set_speed(SIMULATION.speed_idx + 1)
        elif direction == "down":
            SIMULATION.set_speed(SIMULATION.speed_idx - 1)
        elif isinstance(direction, int):
            SIMULATION.set_speed(direction)

    elif msg_type == "pause_toggle":
        IS_PAUSED = not IS_PAUSED

    elif msg_type == "toggle_ultra":
        SIMULATION.ultra_mode = not SIMULATION.ultra_mode

    elif msg_type == "save_full_state":
        filename = data.get("filename")
        notes = data.get("notes", "")
        saved_file = SIMULATION.save_full_state(filename=filename, notes=notes)
        await websocket.send(json.dumps({"type": "save_result", "ok": True, "path": saved_file}))

    elif msg_type == "print_population":
        SIMULATION._print_metric_recap()

    elif msg_type == "load_save_data":
        save_content = data.get("content")
        if save_content:
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir='saves') as f:
                json.dump(save_content, f)
                temp_path = f.name
            result = SIMULATION.load_from_save(temp_path)
            os.unlink(temp_path)
            if result is not None:
                await websocket.send(json.dumps({"type": "load_result", "ok": True, "message": "Simulation loaded from uploaded save"}))
            else:
                await websocket.send(json.dumps({"type": "load_result", "ok": False, "message": "Failed to load save — check file format"}))

    elif msg_type == "reload_with_changes":
        import sys, os
        try:
            save_path = os.path.join("saves", "_reload_state.json")
            SIMULATION.save_full_state(filename="_reload_state", notes="auto-reload")

            await websocket.send(json.dumps({
                "type": "reload_starting",
                "ok": True,
                "message": "Restarting server with code changes...",
                "save_path": save_path,
            }))
            await asyncio.sleep(0.2)

            # Strip existing --load/--path/-p args to avoid duplication on repeated reloads
            clean_argv = []
            skip_next = False
            for arg in sys.argv:
                if skip_next:
                    skip_next = False
                    continue
                if arg in ("--load", "--path", "-p"):
                    skip_next = True
                    continue
                if arg.startswith("--load=") or arg.startswith("--path="):
                    continue
                clean_argv.append(arg)

            os.execv(sys.executable, [sys.executable] + clean_argv + ["--load", save_path])
        except Exception as exc:
            logger.error(f"Reload failed: {exc}")
            await websocket.send(json.dumps({
                "type": "reload_result",
                "ok": False,
                "message": f"The code update failed, try manual saving and reload",
            }))

async def ws_handler(websocket: websockets.WebSocketServerProtocol, path: str = "/") -> None:
    global RELOAD_RESULT
    CLIENTS.add(websocket)
    logger.info(f"Client connected: {websocket.remote_address}")

    # Push pending reload result to the reconnecting client
    if RELOAD_RESULT is not None:
        try:
            await websocket.send(json.dumps(RELOAD_RESULT))
        except Exception:
            pass
        RELOAD_RESULT = None

    try:
        async for message in websocket:
            await handle_client_message(websocket, message)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        CLIENTS.remove(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}")


async def simulation_loop() -> None:
    """Async loop running simulation physics step at real-time speeds."""
    global SIMULATION, IS_PAUSED
    base_dt = 1.0 / 60.0

    while True:
        if SIMULATION is not None and SIMULATION.running and not IS_PAUSED:
            dt = base_dt * SIMULATION.speed_multiplier
            max_step = 0.1
            remaining = dt
            while remaining > 0 and SIMULATION.running:
                step_dt = min(remaining, max_step)
                SIMULATION.step(step_dt)
                remaining -= step_dt
        
        await asyncio.sleep(base_dt)


async def broadcast_loop() -> None:
    """Async loop broadcasting serialized snapshots to connected WebSocket clients."""
    global SIMULATION, IS_PAUSED

    while True:
        from core.constants import BROADCAST_INTERVAL
        interval = float(BROADCAST_INTERVAL)
        if CLIENTS and SIMULATION is not None:
            if SIMULATION.ultra_mode:
                snapshot = build_aggregate_snapshot(SIMULATION.world, SIMULATION, IS_PAUSED)
            else:
                snapshot = build_full_snapshot(SIMULATION.world, SIMULATION, IS_PAUSED)

            message = encode(snapshot)
            # Broadcast concurrently to all connected clients
            websockets.broadcast(CLIENTS, message)

        await asyncio.sleep(interval)


RELOAD_RESULT: dict | None = None


def run_server(host: str = "0.0.0.0", port: int = 8765, load_path: str | None = None) -> None:
    global SIMULATION, RELOAD_RESULT
    import os

    is_reload = (load_path is not None and load_path.endswith("_reload_state.json"))
    old_constants_snapshot = None

    if is_reload and os.path.exists(load_path):
        # Read the saved constants snapshot before we load (and potentially overwrite)
        try:
            with open(load_path, "r") as f:
                import json as _json
                saved = _json.load(f)
                old_constants_snapshot = saved.get("constants_snapshot", {})
        except Exception:
            old_constants_snapshot = None

    SIMULATION = Simulation(load_path=load_path)

    if is_reload and old_constants_snapshot is not None:
        # Compare old (saved) constants against current (freshly-imported) constants
        current_constants = SIMULATION._build_constants_snapshot()
        changed = []
        for k, old_val in old_constants_snapshot.items():
            new_val = current_constants.get(k)
            if new_val is not None and new_val != old_val:
                changed.append(k)
        # Check for newly added constants
        for k in current_constants:
            if k not in old_constants_snapshot:
                changed.append(k)

        if changed:
            RELOAD_RESULT = {
                "type": "reload_result", "ok": True,
                "message": f"Successfully updated the constants {', '.join(changed)}",
            }
        else:
            RELOAD_RESULT = {
                "type": "reload_result", "ok": True,
                "message": "Server restarted — no constant changes detected",
            }
        # Clean up the temp reload save
        try:
            os.unlink(load_path)
        except OSError:
            pass

    http_port = start_http_server(host, port)
    logger.info(f"WebSocket Server starting at ws://{host}:{port}")

    async def main_async():
        async with websockets.serve(ws_handler, host, port):
            await asyncio.gather(
                simulation_loop(),
                broadcast_loop(),
            )

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Server shut down cleanly.")


if __name__ == "__main__":
    run_server()

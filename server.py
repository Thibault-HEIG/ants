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
            try:
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
            except Exception as exc:
                logger.error(f"Load failed: {exc}")
                await websocket.send(json.dumps({"type": "load_result", "ok": False, "message": f"Load failed: {exc}"}))

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

    # Force full training history resend for the new client
    if SIMULATION is not None:
        SIMULATION._last_sent_training_idx = 0
        SIMULATION._training_reset_flag = True

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
    """Async loop that attempts to tick at a smooth target FPS (e.g. 30 Hz).
    
    Instead of calculating a full `dt` (which causes visual jumping at 1x speed),
    we accumulate how many `steps` we should run this tick to reach the user's `target_multiplier`.
    If compute takes longer than the tick budget, the actual multiplier is capped.
    Ultra mode ignores the tick budget and processes frames in bulk as fast as possible.
    """
    import time as _time
    from core.constants import FRAMES_PER_DT

    global SIMULATION, IS_PAUSED

    target_fps = 30.0
    tick_budget = 1.0 / target_fps
    frame_dt = 1.0 / FRAMES_PER_DT  # 0.0333s of sim-time per frame
    
    steps_accumulator = 0.0

    while True:
        if SIMULATION is None or not SIMULATION.running or IS_PAUSED:
            await asyncio.sleep(0.05)
            continue

        target = SIMULATION.target_multiplier
        ultra = SIMULATION.ultra_mode

        t0 = _time.monotonic()

        if ultra:
            # Run in massive batches for raw speed, bypassing standard accumulators
            batch_size = max(int(target * FRAMES_PER_DT), 100)
            for _ in range(batch_size):
                if not SIMULATION.running or IS_PAUSED:
                    break
                SIMULATION.step(frame_dt)
            
            compute_time = _time.monotonic() - t0
            SIMULATION.actual_multiplier = round(batch_size / (FRAMES_PER_DT * max(compute_time, 1e-9)), 1)
            SIMULATION.multiplier_capped = False
            # Yield to event loop to allow broadcast and receiving WS messages
            await asyncio.sleep(0)
            continue

        # Normal mode: Add `target` steps to the accumulator per tick
        # (Since we try to run 30 ticks per real second, this yields `target * 30` steps per sec)
        steps_accumulator += target
        steps_this_tick = int(steps_accumulator)
        steps_accumulator -= steps_this_tick

        for _ in range(steps_this_tick):
            if not SIMULATION.running or IS_PAUSED:
                break
            SIMULATION.step(frame_dt)

        compute_time = _time.monotonic() - t0

        if compute_time >= tick_budget:
            # CPU couldn't finish the batch within the 33ms budget (ceiling hit)
            # Throughput = steps computed / (time taken * frames per sim-second)
            if compute_time > 0:
                SIMULATION.actual_multiplier = round(steps_this_tick / (FRAMES_PER_DT * compute_time), 1)
            SIMULATION.multiplier_capped = True
            await asyncio.sleep(0)
        else:
            # We finished early, we are exactly on schedule
            SIMULATION.actual_multiplier = target
            SIMULATION.multiplier_capped = False
            await asyncio.sleep(tick_budget - compute_time)


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
        async with websockets.serve(ws_handler, host, port, max_size=50 * 1024 * 1024):
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

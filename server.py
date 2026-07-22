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

from core.live_config import LiveConfig
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

    if msg_type == "set_constant":
        key = data.get("key")
        value = data.get("value")
        confirm = bool(data.get("confirm_reset", False))
        res = LiveConfig().set(key, value, confirm_reset=confirm)
        
        # Send confirmation response back to sending client
        response = {"type": "constant_update_result", **res}
        await websocket.send(json.dumps(response))

    elif msg_type == "apply_restart":
        applied_keys = LiveConfig().apply_pending_restart()
        SIMULATION.reset()
        response = {"type": "restart_applied", "keys": applied_keys}
        await websocket.send(json.dumps(response))

    elif msg_type == "set_speed":
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

    elif msg_type == "save_brains":
        saved_file = SIMULATION.save_top_brains()
        await websocket.send(json.dumps({"type": "info", "message": f"Saved top brains to {saved_file}"}))

    elif msg_type == "print_population":
        SIMULATION._print_metric_recap()

    elif msg_type == "get_constants":
        all_consts = LiveConfig().get_all()
        await websocket.send(json.dumps({"type": "constants_data", "constants": all_consts}))


async def ws_handler(websocket: websockets.WebSocketServerProtocol, path: str = "/") -> None:
    CLIENTS.add(websocket)
    logger.info(f"Client connected: {websocket.remote_address}")
    
    # Send initial constants registry on connect
    all_consts = LiveConfig().get_all()
    await websocket.send(json.dumps({"type": "constants_data", "constants": all_consts}))

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
        interval = float(LiveConfig().get("BROADCAST_INTERVAL") or 0.05)
        if CLIENTS and SIMULATION is not None:
            if SIMULATION.ultra_mode:
                snapshot = build_aggregate_snapshot(SIMULATION.world, SIMULATION, IS_PAUSED)
            else:
                snapshot = build_full_snapshot(SIMULATION.world, SIMULATION, IS_PAUSED)

            message = encode(snapshot)
            # Broadcast concurrently to all connected clients
            websockets.broadcast(CLIENTS, message)

        await asyncio.sleep(interval)


def run_server(host: str = "0.0.0.0", port: int = 8765, load_path: str | None = None) -> None:
    global SIMULATION
    SIMULATION = Simulation(load_path=load_path)
    
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

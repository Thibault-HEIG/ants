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


TRACKING_DB: Any | None = None
CURRENT_RUN_ID: int | None = None

class StaticFileHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static files from the project directory (web/ and assets/) and API."""

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self.handle_api_request()
            return
        super().do_GET()

    def handle_api_request(self) -> None:
        global TRACKING_DB
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        if TRACKING_DB is None:
            self.wfile.write(b"[]")
            return
            
        try:
            parts = self.path.split("?")[0].strip("/").split("/")
            if len(parts) == 2 and parts[1] == "runs":
                cursor = TRACKING_DB.conn.execute("SELECT id, started_at, ended_at, notes FROM runs ORDER BY id DESC")
                data = [{"id": r[0], "started_at": r[1], "ended_at": r[2], "notes": r[3]} for r in cursor.fetchall()]
                self.wfile.write(json.dumps(data).encode("utf-8"))
            elif len(parts) >= 3 and parts[1] == "runs":
                run_id = int(parts[2])
                endpoint = parts[3] if len(parts) > 3 else None
                
                if endpoint == "stats":
                    cursor = TRACKING_DB.conn.execute("""
                        SELECT s.time, l.species_name, l.alive, l.max_pop, 
                               l.fitness_best, l.fitness_avg, l.lifetime_best, l.lifetime_avg,
                               l.food_best, l.food_avg, l.enemies_best, l.enemies_avg,
                               l.tiles_best, l.tiles_avg, l.release_home_best, l.release_home_avg
                        FROM live_stats l JOIN snapshots s ON l.snapshot_id = s.id
                        WHERE s.run_id = ? ORDER BY s.time ASC
                    """, (run_id,))
                    cols = [c[0] for c in cursor.description]
                    data = [dict(zip(cols, row)) for row in cursor.fetchall()]
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                elif endpoint == "training":
                    cursor = TRACKING_DB.conn.execute("""
                        SELECT s.time, t.species_name, t.generation, t.metric, 
                               t.best, t.avg, t.best_lifetime, t.avg_lifetime
                        FROM training_metrics t JOIN snapshots s ON t.snapshot_id = s.id
                        WHERE s.run_id = ? ORDER BY s.time ASC
                    """, (run_id,))
                    cols = [c[0] for c in cursor.description]
                    data = [dict(zip(cols, row)) for row in cursor.fetchall()]
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                elif endpoint == "bounds":
                    cursor = TRACKING_DB.conn.execute("""
                        SELECT b.species_name, b.metric, b.max_observed, b.bound
                        FROM metric_bounds b JOIN snapshots s ON b.snapshot_id = s.id
                        WHERE s.run_id = ? AND s.id = (SELECT id FROM snapshots WHERE run_id = ? ORDER BY time DESC LIMIT 1)
                    """, (run_id, run_id))
                    cols = [c[0] for c in cursor.description]
                    data = [dict(zip(cols, row)) for row in cursor.fetchall()]
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                elif endpoint == "creatures":
                    cursor = TRACKING_DB.conn.execute("""
                        SELECT c.lifetime as time, c.species_name, c.creature_uid, c.fitness, c.lifetime, c.food_eaten
                        FROM creatures c
                        WHERE c.run_id = ? ORDER BY c.creature_uid ASC
                    """, (run_id,))
                    cols = [c[0] for c in cursor.description]
                    data = [dict(zip(cols, row)) for row in cursor.fetchall()]
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                elif endpoint == "genomes":
                    cursor = TRACKING_DB.conn.execute("""
                        SELECT s.time, g.species_name, g.generation, g.rank, g.fitness
                        FROM genomes g JOIN snapshots s ON g.snapshot_id = s.id
                        WHERE s.run_id = ? ORDER BY s.time ASC
                    """, (run_id,))
                    cols = [c[0] for c in cursor.description]
                    data = [dict(zip(cols, row)) for row in cursor.fetchall()]
                    self.wfile.write(json.dumps(data).encode("utf-8"))
                else:
                    self.wfile.write(b"[]")
            else:
                self.wfile.write(b"[]")
        except Exception as e:
            logger.error(f"API Error: {e}")
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

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



    elif msg_type == "load_from_db":
        run_id = data.get("run_id")
        reset_stats = data.get("reset_stats", False)
        if run_id:
            try:
                SIMULATION.load_from_db(TRACKING_DB, run_id, reset_stats=reset_stats)
                global CURRENT_RUN_ID
                if reset_stats:
                    TRACKING_DB.end_run(CURRENT_RUN_ID)
                    CURRENT_RUN_ID = TRACKING_DB.start_run(notes=f"Branched from run {run_id}")
                else:
                    CURRENT_RUN_ID = run_id
                    
                SIMULATION.world.on_generation_end = lambda cls: TRACKING_DB.write_genomes(SIMULATION.world, CURRENT_RUN_ID, cls)
                await websocket.send(json.dumps({"type": "load_result", "ok": True, "message": f"Loaded DB run {run_id}"}))
            except Exception as exc:
                logger.error(f"DB Load failed: {exc}")
                await websocket.send(json.dumps({"type": "load_result", "ok": False, "message": f"DB Load failed: {exc}"}))

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


async def stats_write_loop() -> None:
    """Async loop writing snapshots to tracking DB on STATS_INTERVAL cadence."""
    global SIMULATION, TRACKING_DB, CURRENT_RUN_ID, IS_PAUSED
    from core.constants import STATS_INTERVAL

    interval = float(STATS_INTERVAL)
    while True:
        if SIMULATION is not None and SIMULATION.running and not IS_PAUSED and CURRENT_RUN_ID is not None:
            try:
                await asyncio.to_thread(TRACKING_DB.write_snapshot, SIMULATION.world, SIMULATION, CURRENT_RUN_ID)
            except Exception as e:
                logger.error(f"Error writing to tracking DB: {e}")
        await asyncio.sleep(interval)


RELOAD_RESULT: dict | None = None


def run_server(host: str = "0.0.0.0", port: int = 8765, load_path: str | None = None) -> None:
    global SIMULATION, RELOAD_RESULT, TRACKING_DB, CURRENT_RUN_ID
    import os
    from core.tracking_db import TrackingDB

    is_reload = (load_path is not None and load_path.endswith("_reload_state.json"))
    old_constants_snapshot = None
    
    TRACKING_DB = TrackingDB()
    CURRENT_RUN_ID = TRACKING_DB.start_run(notes="Standard run" if not is_reload else "Reloaded run")

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
            
    SIMULATION.world.on_generation_end = lambda cls: TRACKING_DB.write_genomes(SIMULATION.world, CURRENT_RUN_ID, cls)

    http_port = start_http_server(host, port)
    logger.info(f"WebSocket Server starting at ws://{host}:{port}")

    async def main_async():
        async with websockets.serve(ws_handler, host, port, max_size=50 * 1024 * 1024):
            await asyncio.gather(
                simulation_loop(),
                broadcast_loop(),
                stats_write_loop(),
            )

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Server shut down cleanly.")
    finally:
        if CURRENT_RUN_ID is not None and TRACKING_DB is not None:
            TRACKING_DB.end_run(CURRENT_RUN_ID)
            TRACKING_DB.close()

if __name__ == "__main__":
    run_server()

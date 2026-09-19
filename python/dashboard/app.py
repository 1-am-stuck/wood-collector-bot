#!/usr/bin/env python3
"""Local FastAPI dashboard: fly senses (left) + Minecraft eye (right)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.hub import history, latest, pose, publish, topology, voxels
from load_env import load_repo_env
from sense.glossary import glossary

load_repo_env()

STATIC = Path(__file__).resolve().parent / "static"
ROOT = Path(__file__).resolve().parents[2]
RETINA_COLUMNS = ROOT / "configs" / "sense" / "retina_columns.json"

app = FastAPI(title="FruitFly dashboard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/latest")
def api_latest():
    return latest() or {}


@app.get("/api/history")
def api_history():
    return {"ticks": history()}


@app.get("/api/glossary")
def api_glossary():
    """What each channel is in the fly, and what excites it in Minecraft."""
    return glossary()


@app.get("/api/world")
def api_world():
    """Current geometry and pose, for a viewer that just connected."""
    snapshot, seq = voxels()
    return {"pose": pose(), "voxels": snapshot, "seq": seq}


@app.get("/api/retina")
def api_retina():
    """Measured ommatidial directions the fly actually casts.

    The HTML used to draw an 8×4 grid per eye. That grid does not exist: the rays
    come from `column_directions.csv` via farthest-point sampling. Serving the
    generated config lets the retina pane place each luminance sample where that
    ommatidium actually looks.
    """
    if not RETINA_COLUMNS.exists():
        return {"rays": [], "count": 0}
    return json.loads(RETINA_COLUMNS.read_text())


@app.get("/api/topology")
def api_topology():
    """The graph: neuron names, groups, body parts, action wiring.

    Fetched once. On the full connectome the names alone are over two megabytes, so
    sending them with every tick would spend the whole socket budget restating a
    structure that by definition cannot change during a run.
    """
    return topology() or {}


@app.post("/ingest")
async def ingest(tick: dict):
    publish(tick)
    return {"ok": True}


STREAM_HZ = 30


@app.websocket("/ws")
async def ws(sock: WebSocket):
    """Pose every frame, policy tick and geometry only when they change."""
    await sock.accept()
    last_tick_t = None
    last_seq = -1
    sent_topology = False
    period = 1.0 / STREAM_HZ
    try:
        while True:
            payload: dict = {"type": "view", "pose": pose()}
            if not sent_topology:
                graph = topology()
                if graph is not None:
                    payload["topology"] = graph
                    sent_topology = True
            tick = latest()
            t = None if tick is None else tick.get("t")
            if tick is not None and t != last_tick_t:
                payload["tick"] = tick
                last_tick_t = t
            snapshot, seq = voxels()
            if snapshot is not None and seq != last_seq:
                payload["voxels"] = snapshot
                payload["seq"] = seq
                last_seq = seq
            await sock.send_json(payload)
            await asyncio.sleep(period)
    except WebSocketDisconnect:
        return
    except RuntimeError:
        # Socket closed mid-send; nothing to clean up.
        return


def run(host: str = "127.0.0.1", port: int = 8766) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()

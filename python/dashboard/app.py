#!/usr/bin/env python3
"""Local FastAPI dashboard: fly senses (left) + Minecraft eye (right)."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dashboard.hub import history, latest, publish
from load_env import load_repo_env

load_repo_env()

STATIC = Path(__file__).resolve().parent / "static"

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


@app.get("/api/viewer")
def api_viewer():
    try:
        with socket.create_connection(("127.0.0.1", 3007), timeout=0.25):
            return {"up": True, "url": "http://127.0.0.1:3007/"}
    except OSError:
        return {"up": False, "url": None}


@app.post("/ingest")
async def ingest(tick: dict):
    publish(tick)
    return {"ok": True}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    last_t = None
    try:
        while True:
            tick = latest()
            t = None if tick is None else tick.get("t")
            if tick is not None and t != last_t:
                await sock.send_json(tick)
                last_t = t
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return


def run(host: str = "127.0.0.1", port: int = 8766) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
Persistent storage for the vimin-core Center Node.

Agent registrations, the task queue, and task history survive center node
restarts. Uses stdlib sqlite3 with WAL mode — no extra dependencies required.

Design:
  • In-memory dicts/lists remain the hot-path source of truth.
  • DB is written on every mutation (fire-and-forget via asyncio.create_task).
  • DB is read only at startup to restore in-memory state.
  • Full JSON blobs are stored to avoid migration headaches in v1.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = str(Path.home() / ".vimin" / "state.db")

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS task_history (
    id              TEXT PRIMARY KEY,
    agent_id        TEXT,
    task_type       TEXT,
    model_id        TEXT,
    success         INTEGER,
    execution_time_ms REAL,
    submitted_by    TEXT,
    submitted_at    TEXT,
    completed_at    TEXT,
    record_json     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_history_agent ON task_history(agent_id);
CREATE INDEX IF NOT EXISTS idx_task_history_at    ON task_history(submitted_at);

CREATE TABLE IF NOT EXISTS task_queue (
    id              TEXT PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'queued',
    assigned_agent  TEXT,
    submitted_at    TEXT,
    record_json     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id        TEXT PRIMARY KEY,
    registered_at   TEXT,
    last_heartbeat  TEXT,
    status          TEXT NOT NULL DEFAULT 'online',
    loaded_model_id TEXT,
    record_json     TEXT NOT NULL
);
"""


def _open(path: str) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection (creates file + dirs if needed)."""
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    os.chmod(dir_path, 0o700)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL; faster than FULL
    # Restrict DB file to owner-only after creation
    if os.path.exists(path):
        os.chmod(path, 0o600)
    return conn


class Database:
    """Async wrapper around a SQLite database for CenterNode persistence."""

    def __init__(self, path: str = _DB_PATH):
        self.path = path

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Create tables (idempotent). Called once at startup."""
        await asyncio.to_thread(self._init_sync)
        logger.info(f"Database ready: {self.path}")

    def _init_sync(self) -> None:
        with _open(self.path) as conn:
            conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Startup restore
    # ------------------------------------------------------------------

    async def load_state(self) -> Dict[str, Any]:
        """
        Load persisted state into a dict with keys:
          agents, task_queue, task_history
        Called once at startup; results are merged into CenterNode's in-memory state.
        """
        return await asyncio.to_thread(self._load_state_sync)

    def _load_state_sync(self) -> Dict[str, Any]:
        with _open(self.path) as conn:
            # Agents — all with heartbeat in the last 24 h
            agents_rows = conn.execute(
                "SELECT record_json FROM agents "
                "WHERE datetime(last_heartbeat) > datetime('now', '-1 day')"
            ).fetchall()
            agents = [json.loads(r["record_json"]) for r in agents_rows]

            # Task queue — active items only; reset assigned→queued so they
            # are re-dispatched (the agent may have lost the command on restart)
            conn.execute(
                "UPDATE task_queue SET status='queued', assigned_agent=NULL "
                "WHERE status='assigned'"
            )
            conn.commit()
            queue_rows = conn.execute(
                "SELECT record_json FROM task_queue WHERE status IN ('queued','running')"
            ).fetchall()
            task_queue = [json.loads(r["record_json"]) for r in queue_rows]
            for t in task_queue:
                if t.get("status") == "assigned":
                    t["status"] = "queued"
                    t["assigned_agent"] = None

            # Task history — last 200 rows, newest first then reversed
            hist_rows = conn.execute(
                "SELECT record_json FROM task_history "
                "ORDER BY submitted_at DESC LIMIT 200"
            ).fetchall()
            task_history = list(reversed([json.loads(r["record_json"]) for r in hist_rows]))

            return {
                "agents": agents,
                "task_queue": task_queue,
                "task_history": task_history,
            }

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    async def upsert_agent(self, agent_info) -> None:
        """Insert or replace an agent record."""
        record = asdict(agent_info) if hasattr(agent_info, "__dataclass_fields__") else dict(agent_info)
        await asyncio.to_thread(self._upsert_agent_sync, agent_info.agent_id, record)

    def _upsert_agent_sync(self, agent_id: str, record: Dict) -> None:
        with _open(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO agents "
                "(agent_id, registered_at, last_heartbeat, status, loaded_model_id, record_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    agent_id,
                    record.get("registered_at"),
                    record.get("last_heartbeat"),
                    record.get("status", "online"),
                    record.get("loaded_model_id"),
                    json.dumps(record),
                ),
            )
            conn.commit()

    async def update_agent_heartbeat(
        self, agent_id: str, ts: str, status: str, loaded_model_id: Optional[str]
    ) -> None:
        await asyncio.to_thread(
            self._update_agent_heartbeat_sync, agent_id, ts, status, loaded_model_id
        )

    def _update_agent_heartbeat_sync(
        self, agent_id: str, ts: str, status: str, loaded_model_id: Optional[str]
    ) -> None:
        with _open(self.path) as conn:
            conn.execute(
                "UPDATE agents SET last_heartbeat=?, status=?, loaded_model_id=?, "
                "record_json=json_patch(record_json, ?) "
                "WHERE agent_id=?",
                (
                    ts,
                    status,
                    loaded_model_id,
                    json.dumps({
                        "last_heartbeat": ts,
                        "status": status,
                        "loaded_model_id": loaded_model_id,
                    }),
                    agent_id,
                ),
            )
            conn.commit()

    async def delete_agent(self, agent_id: str) -> None:
        await asyncio.to_thread(self._delete_agent_sync, agent_id)

    def _delete_agent_sync(self, agent_id: str) -> None:
        with _open(self.path) as conn:
            conn.execute("DELETE FROM agents WHERE agent_id=?", (agent_id,))
            conn.commit()

    # ------------------------------------------------------------------
    # Task queue
    # ------------------------------------------------------------------

    async def save_task_queue(self, task: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._save_task_queue_sync, task)

    def _save_task_queue_sync(self, task: Dict[str, Any]) -> None:
        with _open(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO task_queue "
                "(id, status, assigned_agent, submitted_at, record_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    task["id"],
                    task.get("status", "queued"),
                    task.get("assigned_agent"),
                    task.get("submitted_at"),
                    json.dumps(task),
                ),
            )
            conn.commit()

    async def update_task_queue(
        self, task_id: str, status: str, assigned_agent: Optional[str], task: Dict[str, Any]
    ) -> None:
        await asyncio.to_thread(
            self._update_task_queue_sync, task_id, status, assigned_agent, task
        )

    def _update_task_queue_sync(
        self, task_id: str, status: str, assigned_agent: Optional[str], task: Dict[str, Any]
    ) -> None:
        with _open(self.path) as conn:
            conn.execute(
                "UPDATE task_queue SET status=?, assigned_agent=?, record_json=? WHERE id=?",
                (status, assigned_agent, json.dumps(task), task_id),
            )
            conn.commit()

    async def remove_from_queue(self, task_id: str) -> None:
        await asyncio.to_thread(self._remove_from_queue_sync, task_id)

    def _remove_from_queue_sync(self, task_id: str) -> None:
        with _open(self.path) as conn:
            conn.execute("DELETE FROM task_queue WHERE id=?", (task_id,))
            conn.commit()

    async def clear_task_queue(self) -> None:
        await asyncio.to_thread(self._clear_task_queue_sync)

    def _clear_task_queue_sync(self) -> None:
        with _open(self.path) as conn:
            conn.execute("DELETE FROM task_queue")
            conn.commit()

    async def remove_tasks_for_agent(self, agent_id: str) -> None:
        await asyncio.to_thread(self._remove_tasks_for_agent_sync, agent_id)

    def _remove_tasks_for_agent_sync(self, agent_id: str) -> None:
        with _open(self.path) as conn:
            rows = conn.execute("SELECT id, record_json FROM task_queue").fetchall()
            for row in rows:
                try:
                    record = json.loads(row["record_json"])
                except Exception:
                    continue
                if record.get("assigned_agent") == agent_id:
                    conn.execute("DELETE FROM task_queue WHERE id=?", (row["id"],))
            conn.commit()

    # ------------------------------------------------------------------
    # Task history
    # ------------------------------------------------------------------

    async def save_task_history(self, record: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._save_task_history_sync, record)

    def _save_task_history_sync(self, record: Dict[str, Any]) -> None:
        with _open(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO task_history "
                "(id, agent_id, task_type, model_id, success, execution_time_ms, "
                " submitted_by, submitted_at, completed_at, record_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.get("task_id") or record.get("id", ""),
                    record.get("agent_id"),
                    record.get("task_type"),
                    record.get("model_id"),
                    1 if record.get("success", True) else 0,
                    record.get("execution_time_ms"),
                    record.get("submitted_by"),
                    record.get("submitted_at"),
                    record.get("completed_at"),
                    json.dumps(record),
                ),
            )
            conn.commit()

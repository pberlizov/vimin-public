"""
Center Node — vimin-core

Lightweight orchestration server for coordinating up to 10 local inference
nodes. Supports broadcast dispatch only — every submitted task goes to all
online nodes simultaneously. Per-node targeting, fleet pipelines, workflows,
and OpenClaw integrations are available in the full vimin distribution.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors

from vimin_core.core.models import ModelRegistry
from vimin_core.core.security import SecurityManager
from vimin_core.core.task import TaskType, TaskComplexity
from vimin_core.systems.db import Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard cap — free/open-source tier
# ---------------------------------------------------------------------------
MAX_NODES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso_timestamp(ts: str) -> datetime:
    if not ts:
        return datetime.now(timezone.utc)
    if ts.endswith("Z"):
        ts = ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


class _JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, deque):
            return list(o)
        if hasattr(o, "to_dict"):
            return o.to_dict()
        return super().default(o)


def _dumps(obj) -> str:
    return json.dumps(obj, cls=_JSONEncoder)


def _err(code: str, message: str, status: int = 400) -> web.Response:
    return web.Response(
        status=status,
        content_type="application/json",
        text=_dumps({"error": code, "message": message}),
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentInfo:
    agent_id: str
    system_info: Dict[str, Any]
    model_status: List[Dict[str, Any]]
    capabilities: Dict[str, Any]
    registered_at: str
    last_heartbeat: str
    status: str = "online"
    metrics_history: Optional[List[Dict[str, Any]]] = None
    session_key: Optional[str] = None
    loaded_model_id: Optional[str] = None

    def __post_init__(self):
        if self.metrics_history is None:
            self.metrics_history = []


@dataclass
class SystemMetrics:
    timestamp: str
    total_agents: int
    online_agents: int
    total_tasks_processed: int
    avg_latency_ms: float
    error_rate: float
    cpu_usage_avg: float
    memory_usage_avg: float
    npu_available_count: int


# ---------------------------------------------------------------------------
# Center Node
# ---------------------------------------------------------------------------

class CenterNode:
    """
    Orchestration center for up to 10 inference nodes.

    All submitted tasks are broadcast to every online node — there is no
    per-node targeting in this edition. The full vimin distribution adds
    tag-based routing, fleet pipelines, OpenClaw integration, and more.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

        self._fleet_token: Optional[str] = os.environ.get("VIMIN_FLEET_TOKEN")

        self.app = web.Application()
        self.agents: Dict[str, AgentInfo] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self.task_history: List[Dict[str, Any]] = []
        self.websocket_clients: List[web.WebSocketResponse] = []
        self._pending_commands: Dict[str, List[Dict[str, Any]]] = {}
        self._continuous_tasks: Dict[str, str] = {}
        self._task_results: Dict[str, str] = {}

        self._audit_log_path = os.path.join(
            os.path.expanduser("~"), ".vimin", "audit.jsonl"
        )
        os.makedirs(os.path.dirname(self._audit_log_path), exist_ok=True)

        self._data_policy: Dict[str, Any] = {
            "blocked_fields": [],
            "version": 0,
            "updated_at": None,
            "updated_by": None,
        }

        self.db = Database()

        self._rate_window = int(os.environ.get("VIMIN_RATE_WINDOW", "60"))
        self._rate_max = int(os.environ.get("VIMIN_RATE_LIMIT", "60"))
        self._rate_timestamps: Dict[str, deque] = defaultdict(deque)

        self.security = SecurityManager()
        self.model_registry = ModelRegistry()

        self.cleanup_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None

        self._setup_routes()
        self._setup_cors()
        logger.info(f"Center Node (core) initialized on {host}:{port}")

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _setup_routes(self):
        r = self.app.router
        # Auth
        r.add_post("/api/auth/register", self._auth_register)
        # Agents
        r.add_post("/api/agents/register", self._register_agent)
        r.add_post("/api/agents/heartbeat", self._heartbeat)
        r.add_post("/api/agents/metrics", self._receive_metrics)
        r.add_post("/api/agents/task-completion", self._task_completion)
        r.add_post("/api/agents/task-stream", self._handle_task_stream)
        r.add_get("/api/agents", self._list_agents)
        r.add_get("/api/agents/{agent_id}", self._get_agent)
        r.add_get("/api/agents/{agent_id}/pending-commands", self._get_pending_commands)
        r.add_post("/api/agents/{agent_id}/set-model", self._set_agent_model)
        # Tasks
        r.add_post("/api/tasks", self._submit_task)
        r.add_get("/api/tasks", self._get_tasks)
        r.add_get("/api/tasks/{task_id}", self._get_task)
        r.add_post("/api/tasks/{task_id}/stop", self._stop_task)
        # Broadcast dispatch
        r.add_post("/api/broadcast", self._broadcast_task)
        # Metrics & policy
        r.add_get("/api/metrics", self._get_system_metrics)
        r.add_get("/api/policy/data", self._get_data_policy)
        r.add_post("/api/policy/data", self._set_data_policy)
        # Models
        r.add_get("/api/models", self._get_models)
        # Infra
        r.add_get("/ws", self._websocket_handler)
        r.add_get("/health", self._health)
        r.add_get("/", self._serve_dashboard)

    def _setup_cors(self):
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=False,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*",
            )
        })
        for route in list(self.app.router.routes()):
            cors.add(route)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, ssl_context=None):
        await self.db.init()
        await self._load_state_from_db()

        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())

        runner = web.AppRunner(self.app)
        await runner.setup()

        # Silence noisy keepalive errors on macOS
        if not os.environ.get("VIMIN_ENABLE_TCP_KEEPALIVE"):
            try:
                from aiohttp import web_protocol as _wp
                def _safe_keepalive(t):
                    try:
                        return _wp._orig_keepalive(t)
                    except Exception:
                        return None
                if not hasattr(_wp, "_orig_keepalive"):
                    _wp._orig_keepalive = _wp.tcp_keepalive
                    _wp.tcp_keepalive = _safe_keepalive
            except Exception:
                pass

        site = web.TCPSite(runner, self.host, self.port, ssl_context=ssl_context)
        await site.start()

        scheme = "https" if ssl_context else "http"
        print(f"  vimin-core center node running at {scheme}://{self.host}:{self.port}")
        print(f"  Dashboard: {scheme}://{self.host}:{self.port}/")
        print(f"  Node limit: {MAX_NODES}")

    async def stop(self):
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.metrics_task:
            self.metrics_task.cancel()
        for ws in self.websocket_clients:
            await ws.close()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _authenticate(self, request) -> Optional[Dict]:
        key = request.headers.get("Authorization", "")
        if key.startswith("Bearer "):
            key = key[7:]
        else:
            key = request.query.get("api_key", "")
        return self.security.validate_key(key)

    async def _auth_register(self, request):
        auth = self._authenticate(request)
        if not auth or not auth.get("is_master"):
            return _err("forbidden", "Master key required.", 403)
        try:
            data = await request.json()
            new_key = self.security.generate_key(data.get("name", "client"), data.get("role", "agent"))
            return web.json_response({"status": "success", "api_key": new_key})
        except Exception as e:
            logger.error(f"Auth register error: {e}")
            return _err("internal_error", "An internal error occurred.", 500)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self, key: str) -> bool:
        now = time.time()
        window = self._rate_timestamps[key]
        cutoff = now - self._rate_window
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self._rate_max:
            return False
        window.append(now)
        return True

    # ------------------------------------------------------------------
    # Agent endpoints
    # ------------------------------------------------------------------

    async def _register_agent(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)

        reg_key = f"reg:{request.remote}"
        if not self._check_rate_limit(reg_key):
            return _err("rate_limit_exceeded", "Too many registration attempts.", 429)

        try:
            data = await request.json()
            agent_id = data["agent_id"]

            # Fleet token check
            if self._fleet_token:
                if data.get("fleet_token", "") != self._fleet_token:
                    logger.warning(f"Agent {agent_id} rejected: invalid fleet token")
                    return _err("invalid_fleet_token", "Fleet token mismatch.", 403)

            # --- 10-node cap ---
            # Allow re-registration of an existing agent (restarts), but block new ones
            # once the cap is reached.
            if agent_id not in self.agents and len(self.agents) >= MAX_NODES:
                logger.warning(
                    f"Node cap ({MAX_NODES}) reached — rejecting new agent {agent_id}. "
                    "Upgrade to the full vimin distribution for larger fleets."
                )
                return _err(
                    "node_limit_reached",
                    f"This vimin-core center node supports up to {MAX_NODES} nodes. "
                    "Upgrade to vimin for larger fleets.",
                    403,
                )

            agent = AgentInfo(
                agent_id=agent_id,
                system_info=data["system_info"],
                model_status=data["model_status"],
                session_key=data.get("session_key"),
                capabilities=data["capabilities"],
                registered_at=data["timestamp"],
                last_heartbeat=data["timestamp"],
            )
            self.agents[agent_id] = agent
            asyncio.create_task(self.db.upsert_agent(agent))
            logger.info(f"Agent registered: {agent_id} ({len(self.agents)}/{MAX_NODES} nodes)")

            await self._broadcast_update({"type": "agent_registered", "data": asdict(agent)})
            return web.json_response({"status": "success", "agent_id": agent_id})

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return _err("internal_error", "An internal error occurred.", 500)

    async def _heartbeat(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        try:
            data = await request.json()
            agent_id = data["agent_id"]
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.last_heartbeat = data["timestamp"]
                agent.status = "online"
                if "loaded_model_id" in data:
                    agent.loaded_model_id = data["loaded_model_id"]
                asyncio.create_task(self.db.update_agent_heartbeat(
                    agent_id, data["timestamp"], "online", agent.loaded_model_id
                ))
            return web.json_response({"status": "success"})
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            return _err("internal_error", "An internal error occurred.", 500)

    async def _receive_metrics(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        try:
            data = await request.json()
            agent_id = data["agent_id"]
            if agent_id in self.agents:
                history = self.agents[agent_id].metrics_history or []
                history.append(data["metrics"])
                if len(history) > 1000:
                    history.pop(0)
                self.agents[agent_id].metrics_history = history
            return web.json_response({"status": "success"})
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return _err("internal_error", "An internal error occurred.", 500)

    async def _list_agents(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        agents_data = [
            {
                "agent_id": aid,
                "hostname": a.system_info.get("hostname"),
                "platform": a.system_info.get("platform"),
                "npu_available": a.system_info.get("npu_available"),
                "status": a.status,
                "last_heartbeat": a.last_heartbeat,
                "registered_at": a.registered_at,
            }
            for aid, a in self.agents.items()
        ]
        return web.json_response({"agents": agents_data, "node_limit": MAX_NODES})

    async def _get_agent(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        agent_id = request.match_info["agent_id"]
        if agent_id not in self.agents:
            return _err("not_found", f"Agent '{agent_id}' not found", 404)
        agent = self.agents[agent_id]
        latest_metrics = agent.metrics_history[-1] if agent.metrics_history else None
        return web.json_response(
            {"agent_info": asdict(agent), "latest_metrics": latest_metrics},
            dumps=_dumps,
        )

    async def _get_pending_commands(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        agent_id = request.match_info["agent_id"]
        cmds = self._pending_commands.pop(agent_id, [])
        return web.json_response({"commands": cmds}, dumps=_dumps)

    async def _set_agent_model(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        agent_id = request.match_info["agent_id"]
        try:
            data = await request.json()
            model_name = data.get("model")
            if not model_name:
                return _err("missing_field", "'model' is required", 400)
            cmd = {"type": "set_model", "model": model_name, "queued_at": get_utc_iso()}
            self._pending_commands.setdefault(agent_id, []).append(cmd)
            return web.json_response({"status": "success", "queued": cmd})
        except Exception as e:
            logger.error(f"set-model error: {e}")
            return _err("internal_error", "An internal error occurred.", 500)

    # ------------------------------------------------------------------
    # Task endpoints
    # ------------------------------------------------------------------

    async def _submit_task(self, request):
        """Submit a task — assigned to the best available single agent."""
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)

        rate_key = auth.get("name") or auth.get("key", "anonymous")
        if not self._check_rate_limit(rate_key):
            return _err("rate_limit_exceeded", f"Max {self._rate_max} submissions per {self._rate_window}s.", 429)

        try:
            data = await request.json()

            raw_data = data.get("data", "")
            if isinstance(raw_data, str) and len(raw_data.encode()) > 32_768:
                return _err("input_too_large", "Input exceeds 32 KB limit.", 413)

            try:
                task_type = TaskType(data.get("type"))
                complexity = TaskComplexity[data.get("complexity", "medium").upper()]
            except (ValueError, KeyError):
                return _err("invalid_request", "Invalid task type or complexity.", 400)

            task = {
                "id": f"task_{int(time.time() * 1000)}",
                "type": task_type.value,
                "data": data.get("data"),
                "complexity": complexity.value,
                "model_id": data.get("model_id"),
                "metadata": data.get("metadata", {}),
                "submitted_by": auth.get("name"),
                "submitted_at": get_utc_iso(),
                "status": "queued",
                "assigned_agent": None,
            }

            self.task_queue.append(task)
            asyncio.create_task(self.db.save_task_queue(task))
            await self._assign_task(task)
            await self._broadcast_update({"type": "task_submitted", "data": task})

            return web.json_response({"status": "success", "task_id": task["id"]})

        except Exception as e:
            logger.error(f"Task submission error: {e}")
            return _err("internal_error", "An internal error occurred.", 500)

    async def _broadcast_task(self, request):
        """Broadcast a prompt to ALL online nodes simultaneously.

        This is the primary dispatch method in vimin-core. Every online node
        receives the same prompt at the same time. Per-node targeting is not
        supported in this edition.

        Body: {"prompt": str, "model_id": str (optional), "max_tokens": int (optional)}
        """
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)

        rate_key = auth.get("name") or auth.get("key", "anonymous")
        if not self._check_rate_limit(rate_key):
            return _err("rate_limit_exceeded", f"Max {self._rate_max} submissions per {self._rate_window}s.", 429)

        try:
            data = await request.json()
            prompt = data.get("prompt", "").strip()
            if not prompt:
                return _err("missing_field", "'prompt' is required.", 400)

            online_agents = [aid for aid, a in self.agents.items() if a.status == "online"]
            if not online_agents:
                return _err("no_agents", "No online nodes to dispatch to.", 400)

            broadcast_id = f"bcast_{int(time.time() * 1000)}"
            task_ids = []

            for agent_id in online_agents:
                task_id = f"{broadcast_id}_{agent_id[:8]}"
                task = {
                    "id": task_id,
                    "type": TaskType.TEXT_GENERATION.value,
                    "data": prompt,
                    "complexity": "medium",
                    "model_id": data.get("model_id"),
                    "metadata": {
                        k: v for k, v in data.items()
                        if k in ("max_tokens", "temperature", "stream")
                    },
                    "submitted_by": auth.get("name"),
                    "submitted_at": get_utc_iso(),
                    "status": "assigned",
                    "assigned_agent": agent_id,
                    "broadcast_id": broadcast_id,
                }
                self.task_queue.append(task)
                self._pending_commands.setdefault(agent_id, []).append({
                    "type": "run_task",
                    "task": task,
                    "data_policy": self._data_policy,
                })
                task_ids.append(task_id)

            await self._broadcast_update({
                "type": "broadcast_dispatched",
                "data": {"broadcast_id": broadcast_id, "task_ids": task_ids, "node_count": len(online_agents)},
            })

            logger.info(f"Broadcast {broadcast_id} dispatched to {len(online_agents)} nodes")
            return web.json_response({
                "status": "success",
                "broadcast_id": broadcast_id,
                "task_ids": task_ids,
                "dispatched_to": len(online_agents),
            })

        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            return _err("internal_error", "An internal error occurred.", 500)

    async def _get_tasks(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        return web.json_response({"tasks": self.task_queue})

    async def _get_task(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        task_id = request.match_info["task_id"]
        for t in self.task_queue:
            if t.get("id") == task_id:
                return web.json_response(t)
        for t in reversed(self.task_history):
            if (t.get("task_id") or t.get("id")) == task_id:
                return web.json_response(t)
        return _err("not_found", f"Task '{task_id}' not found", 404)

    async def _stop_task(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        task_id = request.match_info["task_id"]
        agent_id = self._continuous_tasks.get(task_id)
        if not agent_id:
            for t in self.task_queue:
                if t.get("id") == task_id:
                    agent_id = t.get("assigned_agent")
                    break
        if not agent_id:
            return _err("not_found", f"Task '{task_id}' not found or not assigned.", 404)
        self._pending_commands.setdefault(agent_id, []).append({"type": "stop_task", "task_id": task_id})
        self._continuous_tasks.pop(task_id, None)
        return web.json_response({"status": "success"})

    async def _task_completion(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        try:
            data = await request.json()
            agent_id = data["agent_id"]
            record = data["task_record"]

            blocked = self._data_policy.get("blocked_fields", [])
            for f in blocked:
                if f in record:
                    record[f] = "[BLOCKED_BY_POLICY]"

            if agent_id in self.agents:
                a = self.agents[agent_id]
                record.setdefault("hostname", a.system_info.get("hostname", "unknown"))
                record.setdefault("platform", a.system_info.get("platform", "unknown"))
                record["agent_id"] = agent_id

            self.task_history.append(record)
            if len(self.task_history) > 200:
                self.task_history.pop(0)
            asyncio.create_task(self.db.save_task_history(record))

            task_id = record.get("task_id") or record.get("id", "")
            if task_id and record.get("result"):
                self._task_results[task_id] = record["result"]
            self.task_queue = [t for t in self.task_queue if t.get("id") != task_id]

            try:
                with open(self._audit_log_path, "a") as af:
                    af.write(_dumps(record) + "\n")
            except Exception as e:
                logger.warning(f"Audit log write failed: {e}")

            await self._broadcast_update({"type": "task_completed", "data": record})
            return web.json_response({"status": "success"})

        except Exception as e:
            logger.error(f"Task completion error: {e}")
            return _err("internal_error", "An internal error occurred.", 500)

    async def _handle_task_stream(self, request):
        try:
            data = await request.json()
            await self._broadcast_update({
                "type": "task_stream",
                "agent_id": data.get("agent_id"),
                "task_id": data.get("task_id"),
                "token_chunk": data.get("token_chunk", ""),
                "final": data.get("final", False),
            })
            return web.json_response({"status": "ok"})
        except Exception as e:
            return _err("internal_error", str(e), 500)

    # ------------------------------------------------------------------
    # Metrics & policy
    # ------------------------------------------------------------------

    async def _get_system_metrics(self, request):
        metrics = self._calculate_system_metrics()
        return web.json_response(asdict(metrics))

    def _calculate_system_metrics(self) -> SystemMetrics:
        online = sum(1 for a in self.agents.values() if a.status == "online")
        npu = sum(1 for a in self.agents.values() if a.system_info.get("npu_available"))
        cpus, mems = [], []
        for a in self.agents.values():
            if a.metrics_history:
                cpus.append(a.metrics_history[-1].get("cpu_usage_percent", 0))
                mems.append(a.metrics_history[-1].get("memory_usage_percent", 0))
        recent = self.task_history[-200:]
        latencies = [t.get("execution_time_ms", 0) for t in recent if t.get("execution_time_ms") is not None]
        return SystemMetrics(
            timestamp=get_utc_iso(),
            total_agents=len(self.agents),
            online_agents=online,
            total_tasks_processed=len(self.task_history),
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            error_rate=sum(1 for t in recent if not t.get("success", True)) / len(recent) if recent else 0.0,
            cpu_usage_avg=sum(cpus) / len(cpus) if cpus else 0.0,
            memory_usage_avg=sum(mems) / len(mems) if mems else 0.0,
            npu_available_count=npu,
        )

    async def _get_data_policy(self, request):
        if not self._authenticate(request):
            return _err("unauthorized", "Valid API key required.", 401)
        return web.json_response(self._data_policy)

    async def _set_data_policy(self, request):
        auth = self._authenticate(request)
        if not auth or not auth.get("is_master"):
            return _err("forbidden", "Master key required.", 403)
        try:
            data = await request.json()
        except Exception:
            return _err("invalid_request", "Request body must be valid JSON.", 400)
        blocked = data.get("blocked_fields", [])
        if not isinstance(blocked, list):
            return _err("invalid_request", "'blocked_fields' must be an array of strings.", 400)
        self._data_policy = {
            "blocked_fields": blocked,
            "version": self._data_policy.get("version", 0) + 1,
            "updated_at": get_utc_iso(),
            "updated_by": auth.get("name", "unknown"),
        }
        return web.json_response({"status": "success", "policy": self._data_policy})

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    async def _get_models(self, request):
        models = self.model_registry.list_models()
        return web.json_response({"models": {n: asdict(m) for n, m in models.items()}})

    # ------------------------------------------------------------------
    # Health & dashboard
    # ------------------------------------------------------------------

    async def _health(self, request):
        online = sum(1 for a in self.agents.values() if a.status == "online")
        return web.json_response({
            "status": "ok",
            "edition": "core",
            "agents": len(self.agents),
            "agents_online": online,
            "node_limit": MAX_NODES,
            "queued_tasks": len(self.task_queue),
        })

    async def _serve_dashboard(self, request):
        return web.Response(
            content_type="text/html",
            text=self._dashboard_html(),
        )

    def _dashboard_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>vimin-core</title>
<style>
  body { font-family: monospace; background: #0d0d0d; color: #e0e0e0; margin: 0; padding: 24px; }
  h1 { color: #fff; font-size: 1.2rem; margin-bottom: 4px; }
  .badge { display: inline-block; background: #1a1a2e; border: 1px solid #333;
           border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; color: #aaa; margin-left: 8px; }
  table { border-collapse: collapse; width: 100%; margin-top: 16px; }
  th, td { text-align: left; padding: 6px 12px; border-bottom: 1px solid #222; font-size: 0.85rem; }
  th { color: #888; font-weight: normal; }
  .online { color: #4ade80; } .offline { color: #f87171; }
  #log { margin-top: 16px; max-height: 240px; overflow-y: auto;
         background: #111; border: 1px solid #222; padding: 8px; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>vimin-core <span class="badge">open-source edition</span></h1>
<p style="color:#666;font-size:0.8rem">
  Upgrade to <a href="https://vimin.ai" style="color:#888">vimin</a> for &gt;10 nodes,
  per-node targeting, fleet pipelines, and OpenClaw integration.
</p>
<div id="status">Loading…</div>
<table id="agents-table">
  <thead><tr><th>Node</th><th>Platform</th><th>NPU</th><th>Status</th><th>Last seen</th></tr></thead>
  <tbody id="agents-body"></tbody>
</table>
<div id="log"></div>
<script>
const token = localStorage.getItem('vimin_token') || '';
const hdr = token ? {'Authorization': 'Bearer ' + token} : {};

async function refresh() {
  try {
    const [health, agents] = await Promise.all([
      fetch('/health').then(r => r.json()),
      fetch('/api/agents', {headers: hdr}).then(r => r.json()),
    ]);
    document.getElementById('status').innerHTML =
      `<span style="color:#4ade80">●</span> online &nbsp;|&nbsp; ` +
      `${health.agents_online}/${health.node_limit} nodes &nbsp;|&nbsp; ` +
      `${health.queued_tasks} queued`;
    const tbody = document.getElementById('agents-body');
    tbody.innerHTML = (agents.agents || []).map(a => `
      <tr>
        <td>${a.hostname || a.agent_id}</td>
        <td>${a.platform || '—'}</td>
        <td>${a.npu_available ? '✓' : '—'}</td>
        <td class="${a.status}">${a.status}</td>
        <td>${new Date(a.last_heartbeat).toLocaleTimeString()}</td>
      </tr>`).join('');
  } catch(e) { document.getElementById('status').textContent = 'connecting…'; }
}

refresh();
setInterval(refresh, 3000);

// Live event log via WebSocket
const ws = new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws${token?'?token='+encodeURIComponent(token):''}`);
const log = document.getElementById('log');
ws.onmessage = e => {
  const d = JSON.parse(e.data);
  const line = document.createElement('div');
  line.style.color = '#888';
  line.textContent = new Date().toLocaleTimeString() + '  ' + JSON.stringify(d);
  log.prepend(line);
  if (log.children.length > 50) log.lastChild.remove();
};
</script>
</body>
</html>"""

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    async def _websocket_handler(self, request):
        token = request.query.get("token")
        if token:
            if token.startswith("Bearer "):
                token = token[7:]
            auth = self.security.validate_key(token)
        else:
            auth = self._authenticate(request)
        if not auth:
            return web.Response(status=401, text="Unauthorized — pass your API key as ?token=<key>")

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websocket_clients.append(ws)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            if ws in self.websocket_clients:
                self.websocket_clients.remove(ws)
        return ws

    async def _broadcast_update(self, data):
        if not self.websocket_clients:
            return
        msg = _dumps(data)
        dead = []
        for ws in self.websocket_clients:
            try:
                await ws.send_str(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.websocket_clients:
                self.websocket_clients.remove(ws)

    # ------------------------------------------------------------------
    # Task assignment (single-task path)
    # ------------------------------------------------------------------

    async def _assign_task(self, task: Dict[str, Any]) -> Optional[str]:
        """Assign a single task to the best available agent."""
        task_type = (task.get("type") or "").upper()
        best_id: Optional[str] = None
        best_score = -1.0

        for agent_id, agent in self.agents.items():
            if agent.status != "online":
                continue
            caps = agent.capabilities or {}
            supported = [t.upper() for t in caps.get("supported_task_types", [])]
            if task_type and supported and task_type not in supported:
                continue

            score = 30.0 if agent.system_info.get("npu_available") else 0.0
            if task.get("model_id") and agent.loaded_model_id == task["model_id"]:
                score += 50.0
            latest = agent.metrics_history[-1] if agent.metrics_history else {}
            score += max(0.0, 50.0 - latest.get("cpu_usage_percent", 50.0))
            score += max(0.0, 50.0 - latest.get("memory_usage_percent", 50.0))

            if score > best_score:
                best_score = score
                best_id = agent_id

        if best_id:
            task["assigned_agent"] = best_id
            task["status"] = "assigned"
            asyncio.create_task(self.db.update_task_queue(task["id"], "assigned", best_id, task))
            self._pending_commands.setdefault(best_id, []).append({
                "type": "run_task",
                "task": task,
                "data_policy": self._data_policy,
            })

        return best_id

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _cleanup_loop(self):
        while True:
            try:
                now = datetime.now(timezone.utc)
                timeout = timedelta(minutes=2)
                for agent_id, agent in self.agents.items():
                    if agent.status == "online":
                        if now - parse_iso_timestamp(agent.last_heartbeat) > timeout:
                            agent.status = "offline"
                            logger.info(f"Agent {agent_id} marked offline (missed heartbeats)")
                            await self._broadcast_update({"type": "agent_offline", "data": {"agent_id": agent_id}})
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
            await asyncio.sleep(30)

    async def _metrics_collection_loop(self):
        while True:
            try:
                metrics = self._calculate_system_metrics()
                await self._broadcast_update({"type": "metrics_update", "data": asdict(metrics)})
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
            await asyncio.sleep(15)

    async def _load_state_from_db(self):
        try:
            state = await self.db.load_state()
            for a in state.get("agents", []):
                agent_id = a.get("agent_id")
                if agent_id:
                    self.agents[agent_id] = AgentInfo(**{
                        k: v for k, v in a.items()
                        if k in AgentInfo.__dataclass_fields__
                    })
            self.task_queue = state.get("task_queue", [])
            self.task_history = deque(state.get("task_history", []), maxlen=500)
            logger.info(
                f"Restored {len(self.agents)} agents, "
                f"{len(self.task_queue)} queued tasks from DB"
            )
        except Exception as e:
            logger.warning(f"Could not restore state from DB: {e}")

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
import re
import secrets
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
    first_seen_at: Optional[str] = None
    status: str = "online"
    metrics_history: Optional[List[Dict[str, Any]]] = None
    session_key: Optional[str] = None
    loaded_model_id: Optional[str] = None
    agent_secret_hash: Optional[str] = None
    revoked_at: Optional[str] = None

    def __post_init__(self):
        if self.metrics_history is None:
            self.metrics_history = []
        if self.first_seen_at is None:
            self.first_seen_at = self.registered_at


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

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        logging.getLogger("aiohttp.access").setLevel(logging.WARNING)

        self._fleet_token: Optional[str] = os.environ.get("VIMIN_FLEET_TOKEN")

        self.app = web.Application()
        self.agents: Dict[str, AgentInfo] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self.task_history: deque = deque(maxlen=500)
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
        r.add_post("/api/agents/{agent_id}/revoke", self._revoke_agent)
        # Tasks
        r.add_post("/api/tasks", self._submit_task)
        r.add_get("/api/tasks", self._get_tasks)
        r.add_post("/api/tasks/clear", self._clear_tasks)
        r.add_get("/api/tasks/{task_id}", self._get_task)
        r.add_post("/api/tasks/{task_id}/stop", self._stop_task)
        # Broadcast dispatch
        r.add_post("/api/broadcast", self._broadcast_task)
        # Pipeline orchestration
        r.add_post("/api/pipeline", self._run_pipeline)
        # Metrics & policy
        r.add_get("/api/metrics", self._get_system_metrics)
        r.add_get("/api/policy/data", self._get_data_policy)
        r.add_post("/api/policy/data", self._set_data_policy)
        # Models
        r.add_get("/api/models", self._get_models)
        # Infra
        r.add_get("/ws", self._websocket_handler)
        r.add_get("/health", self._health)
        r.add_get("/api/health", self._health)
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

    def _extract_agent_secret(self, request, data: Optional[Dict[str, Any]] = None) -> str:
        secret = request.headers.get("X-Vimin-Agent-Secret", "")
        if secret:
            return secret
        if isinstance(data, dict):
            return str(data.get("agent_secret", "") or "")
        return ""

    def _validate_agent_identity(
        self,
        agent_id: str,
        agent_secret: str,
        allow_bootstrap: bool = False,
    ) -> Optional[web.Response]:
        agent = self.agents.get(agent_id)
        if not agent:
            return None if allow_bootstrap else _err("not_found", f"Agent '{agent_id}' not found", 404)
        if agent.status == "revoked" or agent.revoked_at:
            return _err("agent_revoked", f"Agent '{agent_id}' has been revoked.", 403)
        if not agent.agent_secret_hash:
            return None if allow_bootstrap else _err("agent_secret_required", "Agent secret required.", 403)
        if not self.security.verify_secret(agent_secret, agent.agent_secret_hash):
            return _err("invalid_agent_secret", "Agent secret mismatch.", 403)
        return None

    def _agent_task_summary(self, agent_id: str) -> Dict[str, int]:
        queued = sum(1 for t in self.task_queue if t.get("assigned_agent") == agent_id)
        completed = sum(1 for t in self.task_history if t.get("agent_id") == agent_id)
        failed = sum(
            1 for t in self.task_history
            if t.get("agent_id") == agent_id and not t.get("success", True)
        )
        return {
            "queued": queued,
            "completed": completed,
            "failed": failed,
            "received_total": queued + completed,
        }

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
            data = json.loads(await request.read())
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
            data = json.loads(await request.read())
            agent_id = data["agent_id"]
            agent_secret = self._extract_agent_secret(request, data)

            # Fleet token check
            if self._fleet_token:
                if data.get("fleet_token", "") != self._fleet_token:
                    logger.warning(f"Agent {agent_id} rejected: invalid fleet token")
                    return _err("invalid_fleet_token", "Fleet token mismatch.", 403)

            # --- 10-node cap ---
            # Allow re-registration of an existing agent (restarts), but block new ones
            # once the cap is reached. Count only online agents — offline/stale agents
            # restored from DB on startup must not consume slots.
            online_count = sum(1 for a in self.agents.values() if a.status == "online")
            if agent_id not in self.agents and online_count >= MAX_NODES:
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

            issued_secret: Optional[str] = None
            existing = self.agents.get(agent_id)
            if existing:
                identity_error = self._validate_agent_identity(agent_id, agent_secret, allow_bootstrap=False)
                if identity_error:
                    return identity_error
                first_seen_at = existing.first_seen_at
                agent_secret_hash = existing.agent_secret_hash
                revoked_at = existing.revoked_at
            else:
                issued_secret = secrets.token_urlsafe(32)
                first_seen_at = data["timestamp"]
                agent_secret_hash = self.security.hash_secret(issued_secret)
                revoked_at = None

            agent = AgentInfo(
                agent_id=agent_id,
                system_info=data["system_info"],
                model_status=data["model_status"],
                session_key=data.get("session_key"),
                capabilities=data["capabilities"],
                registered_at=data["timestamp"],
                last_heartbeat=data["timestamp"],
                first_seen_at=first_seen_at,
                agent_secret_hash=agent_secret_hash,
                revoked_at=revoked_at,
            )
            self.agents[agent_id] = agent
            asyncio.create_task(self.db.upsert_agent(agent))
            online_now = sum(1 for a in self.agents.values() if a.status == "online")
            logger.info(f"Agent registered: {agent_id} ({online_now}/{MAX_NODES} online nodes)")

            await self._broadcast_update({"type": "agent_registered", "data": asdict(agent)})
            response = {"status": "success", "agent_id": agent_id}
            if issued_secret:
                response["agent_secret"] = issued_secret
            return web.json_response(response)

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return _err("internal_error", "An internal error occurred.", 500)

    async def _heartbeat(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        try:
            data = json.loads(await request.read())
            agent_id = data["agent_id"]
            identity_error = self._validate_agent_identity(
                agent_id, self._extract_agent_secret(request, data)
            )
            if identity_error:
                return identity_error
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                agent.last_heartbeat = data["timestamp"]
                # Allow agents to announce graceful shutdown via status=offline
                agent.status = data.get("status", "online")
                if "loaded_model_id" in data:
                    agent.loaded_model_id = data["loaded_model_id"]
                asyncio.create_task(self.db.update_agent_heartbeat(
                    agent_id, data["timestamp"], agent.status, agent.loaded_model_id
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
            data = json.loads(await request.read())
            agent_id = data["agent_id"]
            identity_error = self._validate_agent_identity(
                agent_id, self._extract_agent_secret(request, data)
            )
            if identity_error:
                return identity_error
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
                "first_seen_at": a.first_seen_at,
                "loaded_model_id": a.loaded_model_id,
                "revoked_at": a.revoked_at,
                "task_summary": self._agent_task_summary(aid),
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
            {
                "agent_info": asdict(agent),
                "latest_metrics": latest_metrics,
                "task_summary": self._agent_task_summary(agent_id),
            },
            dumps=_dumps,
        )

    async def _get_pending_commands(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        agent_id = request.match_info["agent_id"]
        identity_error = self._validate_agent_identity(
            agent_id, self._extract_agent_secret(request)
        )
        if identity_error:
            return identity_error
        cmds = self._pending_commands.pop(agent_id, [])
        if cmds:
            logger.info(f"Delivering {len(cmds)} pending command(s) to agent {agent_id}")
        return web.json_response({"commands": cmds}, dumps=_dumps)

    async def _set_agent_model(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        agent_id = request.match_info["agent_id"]
        try:
            data = json.loads(await request.read())
            model_name = data.get("model")
            if not model_name:
                return _err("missing_field", "'model' is required", 400)
            cmd = {"type": "set_model", "model": model_name, "queued_at": get_utc_iso()}
            self._pending_commands.setdefault(agent_id, []).append(cmd)
            return web.json_response({"status": "success", "queued": cmd})
        except Exception as e:
            logger.error(f"set-model error: {e}")
            return _err("internal_error", "An internal error occurred.", 500)

    async def _revoke_agent(self, request):
        auth = self._authenticate(request)
        if not auth or not auth.get("is_master"):
            return _err("forbidden", "Master key required.", 403)
        agent_id = request.match_info["agent_id"]
        agent = self.agents.get(agent_id)
        if not agent:
            return _err("not_found", f"Agent '{agent_id}' not found", 404)

        revoked_at = get_utc_iso()
        agent.status = "revoked"
        agent.revoked_at = revoked_at
        agent.agent_secret_hash = None
        self._pending_commands.pop(agent_id, None)
        self.task_queue = [t for t in self.task_queue if t.get("assigned_agent") != agent_id]
        await self.db.upsert_agent(agent)
        await self.db.remove_tasks_for_agent(agent_id)
        await self._broadcast_update({
            "type": "agent_revoked",
            "data": {"agent_id": agent_id, "revoked_at": revoked_at},
        })
        return web.json_response({
            "status": "success",
            "agent_id": agent_id,
            "revoked_at": revoked_at,
        })

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
            data = json.loads(await request.read())

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
            data = json.loads(await request.read())
            prompt = data.get("prompt", "").strip()
            if not prompt:
                return _err("missing_field", "'prompt' is required.", 400)

            # Classify agents into three buckets:
            #   alive   — status==online AND heartbeated within the last 90s
            #   offline — explicitly offline (missed heartbeats already detected)
            #   ghost   — status==online but heartbeat is stale (process probably crashed)
            # Alive agents get tasks queued and are waited on.
            # Offline agents get tasks queued (they'll pick them up on reconnect) but
            # are NOT waited on — the broadcast returns without blocking on them.
            # Ghost agents are silently excluded.
            now_dt = datetime.now(timezone.utc)
            alive_threshold = timedelta(seconds=90)
            alive_agents: list = []
            offline_agents: list = []
            for aid, a in self.agents.items():
                age = now_dt - parse_iso_timestamp(a.last_heartbeat)
                if a.status == "online" and age <= alive_threshold:
                    alive_agents.append(aid)
                elif a.status == "offline":
                    offline_agents.append(aid)
                # else: ghost — stale online, skip

            if not alive_agents and not offline_agents:
                return _err("no_agents", "No nodes to dispatch to.", 400)

            broadcast_id = f"bcast_{int(time.time() * 1000)}"
            task_ids_alive: list = []   # waited on
            task_ids_offline: list = [] # queued only

            mode = data.get("mode", "return")   # "return" | "broadcast"
            save_local = (mode == "broadcast")

            for agent_id in alive_agents + offline_agents:
                task_id = f"{broadcast_id}_{agent_id[:8]}"
                task = {
                    "id": task_id,
                    "type": TaskType.TEXT_GENERATION.value,
                    "data": prompt,
                    "complexity": "medium",
                    "model_id": data.get("model_id"),
                    "metadata": {
                        **{k: v for k, v in data.items()
                           if k in ("max_tokens", "temperature", "stream")},
                        **({"save_local": True} if save_local else {}),
                    },
                    "submitted_by": auth.get("name"),
                    "submitted_at": get_utc_iso(),
                    "status": "assigned" if agent_id in alive_agents else "queued",
                    "assigned_agent": agent_id,
                    "broadcast_id": broadcast_id,
                }
                self.task_queue.append(task)
                asyncio.create_task(self.db.save_task_queue(task))
                self._pending_commands.setdefault(agent_id, []).append({
                    "type": "run_task",
                    "task": task,
                    "data_policy": self._data_policy,
                })
                if agent_id in alive_agents:
                    task_ids_alive.append(task_id)
                else:
                    task_ids_offline.append(task_id)

            all_task_ids = task_ids_alive + task_ids_offline

            await self._broadcast_update({
                "type": "broadcast_dispatched",
                "data": {
                    "broadcast_id": broadcast_id,
                    "task_ids": all_task_ids,
                    "node_count": len(alive_agents),
                    "queued_count": len(offline_agents),
                },
            })

            logger.info(
                f"Broadcast {broadcast_id} dispatched to {len(alive_agents)} live node(s), "
                f"{len(offline_agents)} queued for offline node(s)"
            )

            # Determine how long to wait for alive agents to respond.
            # Priority: request body "timeout" > query param "timeout" > default 60s.
            # Offline agents are never waited on — their tasks are queued and will be
            # picked up when they reconnect.  Alive agents that don't respond within
            # the timeout are returned as "in_progress" (not "timeout") because the
            # task is still running on the agent; it just won't come back to this
            # particular CLI call.  The result is stored in task_history when done.
            wait = str(request.rel_url.query.get("wait", "true")).lower() != "false"
            timeout_s = float(
                data.get("timeout")
                or request.rel_url.query.get("timeout")
                or 60
            )

            if not wait:
                return web.json_response({
                    "status": "success",
                    "broadcast_id": broadcast_id,
                    "task_ids": all_task_ids,
                    "dispatched_to": len(alive_agents),
                    "queued_for": len(offline_agents),
                })

            # Poll until all live tasks report completion or the timeout fires.
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                if all(tid in self._task_results for tid in task_ids_alive):
                    break
                await asyncio.sleep(0.25)

            def _build_result(tid, agent_id_hint):
                result_payload = self._task_results.get(tid)
                agent_id_for_task = next(
                    (t.get("assigned_agent") for t in self.task_queue if t.get("id") == tid),
                    None,
                ) or next(
                    (t.get("agent_id") for t in self.task_history
                     if (t.get("task_id") or t.get("id")) == tid),
                    agent_id_hint,
                )
                latency_ms = next(
                    (t.get("execution_time_ms", 0) for t in self.task_history
                     if (t.get("task_id") or t.get("id")) == tid),
                    0,
                )
                if result_payload is None:
                    # Task was dispatched to a live agent but hasn't completed yet.
                    # Return in_progress so the caller knows the task IS running —
                    # the result will appear in task history when the agent finishes.
                    return {
                        "agent_id": agent_id_for_task,
                        "output": None,
                        "in_progress": True,
                        "note": "task is running on agent — result will be stored in task history",
                    }
                if isinstance(result_payload, str) and result_payload.startswith("[error] "):
                    return {"agent_id": agent_id_for_task, "output": None,
                            "error": result_payload[8:], "latency_ms": latency_ms}
                return {"agent_id": agent_id_for_task, "output": result_payload, "latency_ms": latency_ms}

            results = [_build_result(tid, tid[:8]) for tid in task_ids_alive]
            # Offline agents are queued, never timed out.
            for tid, agent_id in zip(task_ids_offline, offline_agents):
                results.append({
                    "agent_id": agent_id,
                    "output": None,
                    "queued": True,
                    "note": "offline — task queued, will execute on reconnect",
                })

            return web.json_response({
                "status": "success",
                "broadcast_id": broadcast_id,
                "results": results,
            })

        except Exception as e:
            logger.error(f"Broadcast error: {e}")
            return _err("internal_error", "An internal error occurred.", 500)

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    @staticmethod
    def _substitute_templates(text: str, outputs: Dict[str, str]) -> str:
        """Replace {{stepN_output}} and {{input}} placeholders with actual values."""
        return re.sub(
            r"\{\{(\w+)\}\}",
            lambda m: outputs.get(m.group(1), m.group(0)),
            text,
        )

    async def _dispatch_pipeline_step(
        self,
        pipeline_id: str,
        step_label: str,
        step_spec: Dict[str, Any],
        resolved_data: str,
        default_model: Optional[str],
        agent_id: str,
        save_local: bool = False,
    ) -> Dict[str, Any]:
        """Push one step to a single agent and wait for its result."""
        task_id = f"{pipeline_id}_{step_label}_{agent_id[:8]}"
        task = {
            "id": task_id,
            "type": step_spec.get("type", TaskType.TEXT_GENERATION.value),
            "data": resolved_data,
            "complexity": step_spec.get("complexity", "medium"),
            "model_id": step_spec.get("model_id") or default_model,
            "metadata": {
                **step_spec.get("metadata", {}),
                **({"save_local": True} if save_local else {}),
            },
            "submitted_by": "pipeline",
            "submitted_at": get_utc_iso(),
            "status": "assigned",
            "assigned_agent": agent_id,
            "pipeline_id": pipeline_id,
        }
        self.task_queue.append(task)
        asyncio.create_task(self.db.save_task_queue(task))
        self._pending_commands.setdefault(agent_id, []).append({
            "type": "run_task",
            "task": task,
            "data_policy": self._data_policy,
        })

        timeout_s = float(step_spec.get("timeout", 300))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if task_id in self._task_results:
                break
            await asyncio.sleep(0.25)

        payload = self._task_results.get(task_id)
        if payload is None:
            return {"agent_id": agent_id, "output": None, "error": "timeout"}
        if isinstance(payload, str) and payload.startswith("[error] "):
            return {"agent_id": agent_id, "output": None, "error": payload[8:]}
        return {"agent_id": agent_id, "output": payload}

    async def _run_pipeline(self, request):
        """
        Execute a multi-step pipeline on the fleet.

        Each step runs sequentially on a single chosen agent. Output from step N
        is available as ``{{stepN_output}}`` in any later step's data field.
        An optional ``{{input}}`` placeholder is replaced with the top-level
        ``input`` field (useful for feeding a document into every step).

        Parallel step groups are expressed as a JSON array inside the steps array.
        Each sub-step in the group runs concurrently on a different agent (or the
        same one if fewer agents are available). Their outputs are joined with
        ``---`` and stored as ``{{stepN_output}}``.

        Body:
          {
            "steps": [ <step> | [<step>, ...], ... ],
            "model_id": "mlx-community/...",   // optional default
            "input": "document text …",        // optional, replaces {{input}}
            "name": "my pipeline"              // optional label
          }

        Step schema:
          {
            "type": "TEXT_GENERATION",   // TaskType value
            "data": "Summarise: {{input}}",
            "model_id": "…",             // overrides top-level default
            "complexity": "medium",
            "metadata": { "max_tokens": 400 },
            "timeout": 300
          }
        """
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)

        try:
            body = json.loads(await request.read())
        except Exception:
            return _err("invalid_json", "Request body must be valid JSON.", 400)

        steps = body.get("steps")
        if not steps or not isinstance(steps, list):
            return _err("missing_field", "'steps' must be a non-empty array.", 400)

        # Alive agents only (heartbeat within last 90 s)
        now_dt = datetime.now(timezone.utc)
        alive_threshold = timedelta(seconds=90)
        alive_agents = [
            aid for aid, a in self.agents.items()
            if a.status == "online"
            and (now_dt - parse_iso_timestamp(a.last_heartbeat)) <= alive_threshold
        ]
        if not alive_agents:
            return _err("no_agents", "No live nodes available.", 400)

        default_model  = body.get("model_id")
        pipeline_input = body.get("input", "")
        pipeline_name  = body.get("name", "pipeline")
        pipeline_id    = f"pipe_{int(time.time() * 1000)}"
        save_local     = body.get("mode", "return") == "broadcast"

        # Seed substitution map with the top-level input
        outputs: Dict[str, str] = {"input": pipeline_input}
        step_results = []

        logger.info(f"Pipeline {pipeline_id} ({pipeline_name}) starting — "
                    f"{len(steps)} step(s), {len(alive_agents)} node(s)")

        try:
            for step_idx, step_spec in enumerate(steps):
                step_num = step_idx + 1

                if isinstance(step_spec, list):
                    # ── Parallel group ──────────────────────────────────────
                    group_results = []
                    for sub_idx, sub_spec in enumerate(step_spec):
                        agent_id = alive_agents[sub_idx % len(alive_agents)]
                        resolved = self._substitute_templates(
                            sub_spec.get("data", ""), outputs
                        )
                        result = await self._dispatch_pipeline_step(
                            pipeline_id, f"s{step_num}p{sub_idx}", sub_spec,
                            resolved, default_model, agent_id,
                            save_local=save_local,
                        )
                        group_results.append(result)

                    outputs[f"step{step_num}_output"] = "\n---\n".join(
                        r["output"] for r in group_results
                        if r.get("output") and not str(r.get("output", "")).startswith("[saved_locally]")
                    )
                    step_results.append({
                        "step": step_num, "parallel": True,
                        "results": group_results,
                        "output": outputs[f"step{step_num}_output"],
                    })

                else:
                    # ── Sequential step ────────────────────────────────────
                    agent_id = alive_agents[0]
                    resolved = self._substitute_templates(
                        step_spec.get("data", ""), outputs
                    )
                    result = await self._dispatch_pipeline_step(
                        pipeline_id, f"s{step_num}", step_spec,
                        resolved, default_model, agent_id,
                        save_local=save_local,
                    )
                    raw_out = result.get("output") or ""
                    # In broadcast mode the result is a file path ref — don't
                    # forward "[saved_locally] /path" as text into the next step.
                    outputs[f"step{step_num}_output"] = (
                        "" if str(raw_out).startswith("[saved_locally]") else raw_out
                    )
                    step_results.append({
                        "step": step_num, "parallel": False,
                        "results": [result],
                        "output": raw_out,
                    })

                logger.info(f"Pipeline {pipeline_id} step {step_num}/{len(steps)} complete")

        except Exception as e:
            logger.error(f"Pipeline {pipeline_id} error at step {step_num}: {e}")
            return _err("pipeline_error", f"Step {step_num} failed: {e}", 500)

        final_output = outputs.get(f"step{len(steps)}_output", "")
        return web.json_response({
            "status": "success",
            "pipeline_id": pipeline_id,
            "name": pipeline_name,
            "steps": step_results,
            "final_output": final_output,
        })

    async def _get_tasks(self, request):
        auth = self._authenticate(request)
        if not auth:
            return _err("unauthorized", "Valid API key required.", 401)
        # Merge pending queue + completed history, newest first
        combined = list(self.task_queue) + list(self.task_history)
        combined.sort(
            key=lambda t: t.get("submitted_at") or t.get("timestamp") or "",
            reverse=True,
        )
        return web.json_response({"tasks": combined}, dumps=_dumps)

    async def _clear_tasks(self, request):
        auth = self._authenticate(request)
        if not auth or not auth.get("is_master"):
            return _err("forbidden", "Master key required.", 403)

        cleared_task_ids = {task.get("id") for task in self.task_queue if task.get("id")}
        cleared_count = len(cleared_task_ids)

        self.task_queue.clear()
        self._task_results = {
            task_id: result
            for task_id, result in self._task_results.items()
            if task_id not in cleared_task_ids
        }

        for agent_id, commands in list(self._pending_commands.items()):
            retained = []
            for command in commands:
                task = command.get("task") if isinstance(command, dict) else None
                task_id = task.get("id") if isinstance(task, dict) else None
                if command.get("type") == "run_task" and task_id in cleared_task_ids:
                    continue
                retained.append(command)
            if retained:
                self._pending_commands[agent_id] = retained
            else:
                self._pending_commands.pop(agent_id, None)

        await self.db.clear_task_queue()
        await self._broadcast_update({
            "type": "tasks_cleared",
            "data": {"cleared_count": cleared_count},
        })
        return web.json_response({
            "status": "success",
            "cleared_count": cleared_count,
            "note": "Queued commands were cleared. Already-running tasks on agents are not interrupted.",
        })

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
            data = json.loads(await request.read())
            agent_id = data["agent_id"]
            identity_error = self._validate_agent_identity(
                agent_id, self._extract_agent_secret(request, data)
            )
            if identity_error:
                return identity_error
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
            asyncio.create_task(self.db.save_task_history(record))

            task_id = record.get("task_id") or record.get("id", "")
            if task_id:
                # If the task failed and carries an error message, surface it.
                # Otherwise store whatever result we got (including empty string
                # for a successful-but-empty model output).
                if not record.get("success", True) and record.get("error"):
                    self._task_results[task_id] = f"[error] {record['error']}"
                else:
                    self._task_results[task_id] = record.get("result", "")
                logger.debug(f"Task {task_id} result stored: {str(self._task_results[task_id])[:120]!r}")
            self.task_queue = [t for t in self.task_queue if t.get("id") != task_id]
            if task_id:
                asyncio.create_task(self.db.remove_from_queue(task_id))
            logger.info(
                f"Task completion received: task={task_id or 'unknown'} agent={agent_id} "
                f"success={record.get('success', True)}"
            )

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
        if not self._authenticate(request):
            return _err("unauthorized", "Valid API key required.", 401)
        try:
            data = json.loads(await request.read())
            identity_error = self._validate_agent_identity(
                data.get("agent_id", ""), self._extract_agent_secret(request, data)
            )
            if identity_error:
                return identity_error
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
        if not self._authenticate(request):
            return _err("unauthorized", "Valid API key required.", 401)
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
        recent = list(self.task_history)[-200:]
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
            data = json.loads(await request.read())
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
  .online { color: #4ade80; } .offline { color: #f87171; } .revoked { color: #f59e0b; }
  #log { margin-top: 16px; max-height: 240px; overflow-y: auto;
         background: #111; border: 1px solid #222; padding: 8px; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>vimin-core <span class="badge">source-available edition</span></h1>
<p style="color:#666;font-size:0.8rem">
  Upgrade to <a href="https://viminlabs.com" style="color:#888">vimin</a> for &gt;10 nodes,
  per-node targeting, fleet pipelines, and OpenClaw integration.
</p>
<div id="status">Loading…</div>
<table id="agents-table">
  <thead><tr><th>Node</th><th>Platform</th><th>NPU</th><th>Status</th><th>Joined</th><th>Tasks</th><th>Last seen</th></tr></thead>
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
        <td>${a.first_seen_at ? new Date(a.first_seen_at).toLocaleString() : '—'}</td>
        <td>${(a.task_summary && a.task_summary.received_total) || 0}</td>
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
                ghost_timeout = timedelta(minutes=2)
                purge_timeout = timedelta(minutes=10)

                to_purge = []
                for agent_id, agent in list(self.agents.items()):
                    age = now - parse_iso_timestamp(agent.last_heartbeat)
                    if agent.status == "online" and age > ghost_timeout:
                        agent.status = "offline"
                        logger.info(f"Agent {agent_id} marked offline (missed heartbeats)")
                        await self._broadcast_update({"type": "agent_offline", "data": {"agent_id": agent_id}})
                    elif agent.status == "offline" and age > purge_timeout:
                        to_purge.append(agent_id)

                for agent_id in to_purge:
                    del self.agents[agent_id]
                    self._pending_commands.pop(agent_id, None)
                    logger.info(f"Purged stale agent {agent_id} (offline >10 min)")
                    await self._broadcast_update({"type": "agent_purged", "data": {"agent_id": agent_id}})

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
            # Mark ALL restored agents offline — only a fresh registration proves
            # an agent is currently running. This prevents stale agents from
            # receiving broadcast tasks after a center node restart.
            for agent in self.agents.values():
                agent.status = "offline"
            # Rebuild _pending_commands for tasks that were queued for specific agents
            # before the center restarted. When those agents reconnect they will poll
            # and receive their pending work automatically.
            for task in self.task_queue:
                aid = task.get("assigned_agent")
                if aid and task.get("status") == "queued":
                    self._pending_commands.setdefault(aid, []).append({
                        "type": "run_task",
                        "task": task,
                        "data_policy": self._data_policy,
                    })
                    logger.info(
                        f"Rebuilt queued task for reconnect delivery: task={task.get('id')} agent={aid}"
                    )
            queued_count = sum(len(v) for v in self._pending_commands.values())
            logger.info(
                f"Restored {len(self.agents)} agents (all marked offline pending re-registration), "
                f"{len(self.task_queue)} queued tasks from DB "
                f"({queued_count} pending commands rebuilt)"
            )
        except Exception as e:
            logger.warning(f"Could not restore state from DB: {e}")

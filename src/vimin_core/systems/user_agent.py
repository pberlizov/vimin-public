#!/usr/bin/env python3
"""
User Agent - Client-side component for NPU Orchestrator System

Manages local model installation, telemetry collection, and communication with center node.
Provides real-time status updates and performance monitoring.
"""

import asyncio
import json
import logging
import os
import platform
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore
    AIOHTTP_AVAILABLE = False
import psutil

try:
    from zeroconf import Zeroconf, ServiceBrowser, ServiceListener
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False

from vimin_core.cli.config import load_config, save_config
from vimin_core.hardware.telemetry import TelemetryCollector
from vimin_core.core.task import Task, TaskType, TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """System information for the user agent"""
    agent_id: str
    hostname: str
    platform: str
    architecture: str
    python_version: str
    npu_available: bool
    total_memory_gb: float
    cpu_cores: int
    gpu_info: Optional[str] = None


@dataclass
class ModelStatus:
    """Status of installed models"""
    model_name: str
    model_path: str
    is_installed: bool
    file_size_mb: float
    last_updated: Optional[str] = None
    version: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics from the user agent"""
    timestamp: str
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_gb: float
    battery_percent: Optional[float]
    thermal_state: Optional[str]
    active_inferences: int
    total_tasks_processed: int
    avg_latency_ms: float
    error_rate: float


class UserAgent:
    """
    User Agent - Manages local NPU orchestration and communicates with center node
    """

    def __init__(self, center_node_url: str = "http://localhost:8080", agent_id: Optional[str] = None, api_key: Optional[str] = None, privacy_mode: bool = False, tls_ca: Optional[str] = None, tls_verify: bool = True, fleet_token: Optional[str] = None, openclaw_url: Optional[str] = None):
        # VIMIN_CENTER_URL env var overrides the default so agents can target a
        # publicly-deployed center node without any code changes.
        self.center_node_url = os.environ.get("VIMIN_CENTER_URL", center_node_url)
        self.agent_id = agent_id or str(uuid.uuid4())
        self.api_key = api_key or os.environ.get("ORCHESTRATOR_API_KEY")
        self.privacy_mode = privacy_mode or os.environ.get("ORCHESTRATOR_PRIVACY_MODE", "").lower() == "true"
        self.tls_ca = tls_ca
        self.tls_verify = tls_verify
        self.fleet_token = fleet_token or os.environ.get("VIMIN_FLEET_TOKEN")
        self.openclaw_url = openclaw_url or os.environ.get("OPENCLAW_URL")
        if self.openclaw_url:
            os.environ["OPENCLAW_URL"] = self.openclaw_url
        self.session = None
        self.heartbeat_task = None
        self.metrics_task = None
        self.commands_task = None
        self.running = False
        self.on_command = None  # Callback for commands: async def (command_type, data)
        # Inference lock: MLX (and most backends) are not thread-safe — only one
        # inference may run at a time per agent process.
        self._inference_lock = asyncio.Lock()

        # Core components
        self.orchestrator = None
        self.telemetry = TelemetryCollector()

        # Metrics tracking
        self.metrics_history = []
        self.task_history = []
        # Continuous tasks: task_id → asyncio.Event (set to request stop)
        self._continuous_tasks: Dict[str, asyncio.Event] = {}
        self.model_registry = {}
        self._agent_secret: Optional[str] = load_config().get("agent_secret")

        # Offline resilience: track connectivity; buffer telemetry to disk when
        # the center node is unreachable and replay it on reconnect.
        self._connected: bool = True
        self._offline_buffer_path: str = os.path.join(
            os.path.expanduser("~"), ".vimin", "offline_buffer.ndjson"
        )
        os.makedirs(os.path.dirname(self._offline_buffer_path), exist_ok=True)

        # Payload encryption: generate a per-session Fernet key so the center
        # node can encrypt task data before dispatching it.  The key is sent
        # in the registration payload; if cryptography is not installed the
        # agent still works — tasks will just be sent in plaintext.
        self._loaded_model_id: Optional[str] = None  # Model currently loaded in memory
        self._model_ready = asyncio.Event()  # Set once a generative model is loaded

        # Whisper backend for SPEECH_TO_TEXT tasks (lazy-loaded on first STT task)
        self._whisper_backend = None
        self._whisper_model_id: Optional[str] = None

        self._start_time: float = time.time()
        self._session_key: Optional[str] = None
        self._fernet = None
        try:
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            self._session_key = key.decode()
            self._fernet = Fernet(key)
            logger.debug("Payload encryption enabled (Fernet)")
        except ImportError:
            logger.info(
                "cryptography not installed — task payloads will not be encrypted. "
                "Run: pip install cryptography"
            )

        logger.info(f"User Agent initialized with ID: {self.agent_id}")
    
    async def start(self):
        """Start user agent"""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for UserAgent. Run: pip install aiohttp")
        
        # Auto-discovery via mDNS only when the URL is still localhost AND no
        # explicit internet URL was set via VIMIN_CENTER_URL.
        if not os.environ.get("VIMIN_CENTER_URL") and (
            "localhost" in self.center_node_url or "127.0.0.1" in self.center_node_url
        ):
            discovered = await self._discover_center_node_via_mdns()
            if discovered:
                self.center_node_url = discovered
                logger.debug(f"Auto-discovered Center Node: {self.center_node_url}")

        # ── URL pinning ────────────────────────────────────────────────────────
        # Persist the center URL on first successful connect.  If it changes on
        # a subsequent run, warn the operator so silent redirections are visible.
        try:
            import json as _json
            _cfg_path = os.path.join(os.path.expanduser("~"), ".vimin", "config.json")
            _cfg: dict = {}
            if os.path.exists(_cfg_path):
                try:
                    _cfg = _json.loads(open(_cfg_path).read())
                except Exception:
                    pass
            _pinned = _cfg.get("pinned_center_url")
            if _pinned and _pinned != self.center_node_url:
                print(
                    f"[vimin] WARNING: center URL changed from pinned value.\n"
                    f"  Pinned:  {_pinned}\n"
                    f"  Current: {self.center_node_url}\n"
                    f"  If this is intentional, delete 'pinned_center_url' from ~/.vimin/config.json.",
                    flush=True,
                )
            elif not _pinned:
                _cfg["pinned_center_url"] = self.center_node_url
                os.makedirs(os.path.dirname(_cfg_path), exist_ok=True)
                with open(_cfg_path, "w") as _f:
                    _f.write(_json.dumps(_cfg, indent=2))
        except Exception as _pin_err:
            logger.debug(f"URL pinning check failed (non-fatal): {_pin_err}")

        # ── TLS warning ────────────────────────────────────────────────────────
        _is_loopback = any(
            h in self.center_node_url
            for h in ("localhost", "127.0.0.1", "::1")
        )
        if not _is_loopback and self.center_node_url.startswith("http://"):
            print(
                f"[vimin] WARNING: connecting to {self.center_node_url} over plain HTTP.\n"
                f"  Credentials and task data are transmitted unencrypted.\n"
                f"  Use HTTPS for connections across untrusted networks.",
                flush=True,
            )

        logger.info("Starting User Agent...")
        self.running = True

        # Build SSL connector for HTTPS center nodes
        connector = None
        if self.center_node_url.startswith("https://"):
            import ssl as _ssl
            if not self.tls_verify:
                connector = aiohttp.TCPConnector(ssl=False)
                logger.warning("TLS certificate verification disabled.")
            elif self.tls_ca:
                ssl_ctx = _ssl.create_default_context(cafile=self.tls_ca)
                connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            else:
                connector = aiohttp.TCPConnector(ssl=True)  # system CA bundle
        self.session = aiohttp.ClientSession(connector=connector)
        
        # Wire to a real NPUOrchestrator for local AI execution
        try:
            from vimin_core.core.orchestrator import NPUOrchestrator
            self.orchestrator = NPUOrchestrator()
            logger.info("UserAgent: NPUOrchestrator wired successfully")
        except Exception as e:
            logger.warning(f"UserAgent: Could not initialize NPUOrchestrator: {e}. Falling back to demo mode.")
            self.orchestrator = None
        
        # Register before starting background loops so the center can issue the
        # per-agent secret first. Otherwise heartbeat / polling can start with
        # incomplete credentials and spam 401s.
        registered = await self._register_with_center()
        if not registered:
            self.running = False
            if self.session:
                await self.session.close()
                self.session = None
            raise RuntimeError("Failed to register with center node")

        # Start background tasks only after registration succeeds.
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.commands_task = asyncio.create_task(self._command_polling_loop())
        
        logger.info("User Agent started successfully")
    
    async def stop(self):
        """Stop user agent"""
        logger.info("Stopping User Agent...")
        self.running = False

        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        if self.metrics_task:
            self.metrics_task.cancel()

        if self.commands_task:
            self.commands_task.cancel()

        # Announce graceful shutdown so the center immediately frees the node slot
        try:
            goodbye = {
                "agent_id": self.agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "status": "offline",
                "uptime_seconds": time.time() - self._start_time,
                "loaded_model_id": self._loaded_model_id,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            if self._agent_secret:
                headers["X-Vimin-Agent-Secret"] = self._agent_secret
            self._heartbeat_sync(
                f"{self.center_node_url}/api/agents/heartbeat",
                goodbye,
                headers,
            )
        except Exception as _bye_err:
            logger.debug(f"Goodbye heartbeat failed (non-fatal): {_bye_err}")

        if self.session:
            await self.session.close()

        if self.orchestrator:
            self.orchestrator.cleanup()

        logger.info("User Agent stopped")

    async def _discover_center_node_via_mdns(self, timeout: float = 5.0) -> Optional[str]:
        """Discover center node URL via mDNS"""
        import socket
        if not ZEROCONF_AVAILABLE:
            return None

        logger.info("Discovering Center Node via mDNS...")
        
        class CenterNodeListener(ServiceListener):
            def __init__(self):
                self.discovered_url = None

            def add_service(self, zc, type_, name):
                info = zc.get_service_info(type_, name)
                if info:
                    addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
                    if addresses:
                        # Use first address and port
                        self.discovered_url = f"http://{addresses[0]}:{info.port}"
                        logger.info(f"Discovered Center Node: {name} at {self.discovered_url}")

            def update_service(self, zc, type_, name):
                pass

            def remove_service(self, zc, type_, name):
                pass

        zc = Zeroconf()
        listener = CenterNodeListener()
        browser = ServiceBrowser(zc, "_npu-orch._tcp.local.", listener)
        
        start_time = time.time()
        try:
            while time.time() - start_time < timeout:
                if listener.discovered_url:
                    return listener.discovered_url
                await asyncio.sleep(0.5)
        finally:
            zc.close()
            
        return None
    
    async def _register_with_center(self) -> bool:
        """Register this agent with the center node."""
        try:
            system_info = self._get_system_info()
            model_status = self._get_model_status()
            
            registration_data = {
                "agent_id": self.agent_id,
                "system_info": asdict(system_info),
                "model_status": [asdict(m) for m in model_status],
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "capabilities": self._get_capabilities(),
                "session_key": self._session_key,  # Fernet key for payload encryption (None if unavailable)
                "fleet_token": self.fleet_token,   # Fleet enrollment token (None if open registration)
            }
            if self._agent_secret:
                registration_data["agent_secret"] = self._agent_secret
            
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            
            async with self.session.post(
                f"{self.center_node_url}/api/agents/register",
                json=registration_data,
                headers=headers
            ) as response:
                if response.status == 200:
                    payload = await response.json()
                    issued_secret = payload.get("agent_secret")
                    if issued_secret and issued_secret != self._agent_secret:
                        self._agent_secret = issued_secret
                        cfg = load_config()
                        cfg["agent_secret"] = issued_secret
                        save_config(cfg)
                    logger.info("Successfully registered with center node")
                    return True
                else:
                    logger.error(f"Failed to register: {response.status}")
                    return False
        
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return False
    
    def _heartbeat_sync(self, url: str, data: dict, headers: dict) -> int:
        """Synchronous urllib heartbeat POST. Returns HTTP status code."""
        import urllib.request as _urlreq
        body = json.dumps(data).encode()
        req = _urlreq.Request(
            url,
            data=body,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        with _urlreq.urlopen(req, timeout=10) as resp:
            return resp.status

    async def _heartbeat_loop(self):
        """Send periodic heartbeat to center node. Tracks connectivity state and
        flushes the offline telemetry buffer when the center node becomes reachable
        again after a period of disconnection."""
        while self.running:
            try:
                heartbeat_data = {
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    "status": "online",
                    "uptime_seconds": time.time() - self._start_time,
                    "loaded_model_id": self._loaded_model_id,
                }
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                if self._agent_secret:
                    headers["X-Vimin-Agent-Secret"] = self._agent_secret
                url = f"{self.center_node_url}/api/agents/heartbeat"
                status = await asyncio.to_thread(self._heartbeat_sync, url, heartbeat_data, headers)
                if status == 200:
                    if not self._connected:
                        logger.info("Center node reachable again — flushing offline buffer")
                        self._connected = True
                        asyncio.create_task(self._flush_offline_buffer())
                else:
                    logger.warning(f"Heartbeat failed: {status}")

                await asyncio.sleep(30)

            except Exception as e:
                if self._connected:
                    logger.warning(f"Center node unreachable: {e}. Buffering telemetry locally.")
                    self._connected = False
                await asyncio.sleep(30)
    
    async def _metrics_collection_loop(self):
        """Collect and send performance metrics"""
        while self.running:
            try:
                metrics = self._collect_performance_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 entries
                if len(self.metrics_history) > 1000:
                    self.metrics_history.pop(0)
                
                # Send to center node
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                if self._agent_secret:
                    headers["X-Vimin-Agent-Secret"] = self._agent_secret
                
                async with self.session.post(
                    f"{self.center_node_url}/api/agents/metrics",
                    json={
                        "agent_id": self.agent_id,
                        "metrics": asdict(metrics)
                    },
                    headers=headers
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Metrics send failed: {response.status}")
                
                await asyncio.sleep(60)  # Send metrics every minute
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)

    def _poll_commands_sync(self, url: str, headers: dict) -> dict:
        """Synchronous urllib GET for pending commands. Returns parsed JSON dict."""
        import urllib.request as _urlreq
        req = _urlreq.Request(url, headers=headers, method="GET")
        with _urlreq.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    async def _command_polling_loop(self):
        """Poll for commands from center node"""
        while self.running:
            try:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                if self._agent_secret:
                    headers["X-Vimin-Agent-Secret"] = self._agent_secret
                url = f"{self.center_node_url}/api/agents/{self.agent_id}/pending-commands"

                data = await asyncio.to_thread(self._poll_commands_sync, url, headers)
                commands = data.get('commands', [])
                for cmd in commands:
                    logger.info(f"Received command: {cmd}")

                    cmd_type = cmd.get('type')
                    if cmd_type == 'set_model':
                        model_id = cmd.get('model')
                        if model_id and self.orchestrator:
                            asyncio.create_task(self._load_model_async(model_id))
                        else:
                            logger.info(f"set_model: no model_id or orchestrator not ready")
                    elif cmd_type == 'run_task':
                        task_dict = cmd.get('task', {})
                        data_policy = cmd.get('data_policy', {})
                        logger.info(
                            f"Received queued task dispatch: task={task_dict.get('id') or task_dict.get('task_id')} "
                            f"type={task_dict.get('type')} save_local={task_dict.get('metadata', {}).get('save_local', False)}"
                        )
                        print(
                            f"[vimin] Task dispatch received: {task_dict.get('id') or task_dict.get('task_id')} "
                            f"({task_dict.get('type', 'unknown')})",
                            flush=True,
                        )
                        # Decrypt data field if the center node encrypted it
                        raw_data = task_dict.get('data', '')
                        if task_dict.get('encrypted') and self._fernet and raw_data:
                            try:
                                raw_data = self._fernet.decrypt(raw_data.encode()).decode()
                            except Exception as dec_err:
                                logger.warning(f"Failed to decrypt task data: {dec_err}")
                        task_dict = {**task_dict, 'data': raw_data}

                        if task_dict.get('continuous'):
                            asyncio.create_task(
                                self._run_continuous_task(task_dict, data_policy)
                            )
                        else:
                            metadata = task_dict.get('metadata', {})
                            task = Task(
                                type=TaskType[task_dict.get('type', 'TEXT_GENERATION').upper()],
                                data=raw_data,
                                complexity=TaskComplexity[task_dict.get('complexity', 'low').upper()],
                                id=task_dict.get('task_id', task_dict.get('id', str(uuid.uuid4()))),
                                metadata=metadata,
                            )
                            model_id = task_dict.get('model_id')
                            if model_id and self.orchestrator:
                                asyncio.create_task(self._run_task_with_model(task, model_id))
                            else:
                                asyncio.create_task(self.execute_task(task))

                    elif cmd_type == 'stop_task':
                        task_id = cmd.get('task_id', '')
                        stop_event = self._continuous_tasks.get(task_id)
                        if stop_event:
                            logger.info(f"Stopping continuous task {task_id}")
                            stop_event.set()
                        else:
                            logger.debug(f"stop_task received for unknown task {task_id}")

                    if self.on_command:
                        if asyncio.iscoroutinefunction(self.on_command):
                            await self.on_command(cmd['type'], cmd.get('data', cmd))
                        else:
                            self.on_command(cmd['type'], cmd.get('data', cmd))
                    
                await asyncio.sleep(2)  # Poll every 2 seconds
                
            except Exception as e:
                logger.error(f"Command polling error: {e}")
                await asyncio.sleep(5)
    
    def _get_system_info(self) -> SystemInfo:
        """Get system information"""
        try:
            import GPUtil
            gpu_info = GPUtil.getGPUs()[0].name if GPUtil.getGPUs() else None
        except:
            gpu_info = None
        
        # Get NPU availability from telemetry
        npu_available = self.telemetry._telemetry.get_npu_availability()
        
        return SystemInfo(
            agent_id=self.agent_id,
            hostname=platform.node(),
            platform=platform.system(),
            architecture=platform.machine(),
            python_version=platform.python_version(),
            npu_available=npu_available,
            total_memory_gb=psutil.virtual_memory().total / (1024**3),
            cpu_cores=psutil.cpu_count(),
            gpu_info=gpu_info
        )
    
    def _get_model_status(self) -> List[ModelStatus]:
        """Return the status of the currently loaded generative model, if any."""
        if self._loaded_model_id:
            return [ModelStatus(
                model_name=self._loaded_model_id,
                model_path="",
                is_installed=True,
                file_size_mb=0.0,
            )]
        return []
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        telemetry_data = self.telemetry.get_latest()
        
        # Calculate real avg latency and error rate from task history
        recent_tasks = self.task_history[-100:]  # Use last 100 tasks
        if recent_tasks:
            latencies = [t.get("execution_time_ms", 0) for t in recent_tasks]
            avg_latency = sum(latencies) / len(latencies)
            errors = sum(1 for t in recent_tasks if not t.get("success", True))
            error_rate = errors / len(recent_tasks)
        else:
            avg_latency = 0.0
            error_rate = 0.0
        
        return PerformanceMetrics(
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            cpu_usage_percent=psutil.cpu_percent(),
            memory_usage_percent=psutil.virtual_memory().percent,
            memory_available_gb=psutil.virtual_memory().available / (1024**3),
            battery_percent=telemetry_data.battery_percent,
            thermal_state=str(telemetry_data.thermal_state_celsius) if telemetry_data.thermal_state_celsius else None,
            active_inferences=0,
            total_tasks_processed=len(self.task_history),
            avg_latency_ms=avg_latency,
            error_rate=error_rate
        )
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities"""
        try:
            task_types = [t.value for t in TaskType]
            complexities = [c.value for c in TaskComplexity]
        except:
            # Fallback if TaskType/TaskComplexity not available
            task_types = ["pii_masking", "classification", "text_generation", "speech_to_text"]
            complexities = ["low", "medium", "high"]
        
        try:
            npu_available = self.telemetry._telemetry.get_npu_availability()
        except:
            npu_available = False
        
        return {
            "supported_task_types": task_types,
            "supported_complexities": complexities,
            "npu_available": npu_available,
            "max_concurrent_tasks": 4,  # TODO: Make configurable
            "supported_providers": ["CPU", "CoreML", "CUDA"] if npu_available else ["CPU"]
        }
    
    async def _load_model_async(self, model_id: str) -> bool:
        """Load a generative model in the background without blocking the event loop."""
        logger.info(f"Loading generative model: {model_id}")
        print(f"[vimin] Loading model: {model_id} — this may take a minute on first use ...", flush=True)
        t0 = time.time()
        try:
            loop = asyncio.get_event_loop()
            ok = await loop.run_in_executor(
                None, self.orchestrator.load_generative_model, model_id
            )
            if ok:
                self._loaded_model_id = model_id
                self._model_ready.set()
                elapsed = time.time() - t0
                logger.info(f"Model loaded successfully: {model_id}")
                print(f"[vimin] Model ready: {model_id} ({elapsed:.1f}s)", flush=True)
            else:
                logger.warning(f"Model load returned False: {model_id}")
                print(f"[vimin] WARNING: model load returned False for {model_id}", flush=True)
            return ok
        except Exception as exc:
            logger.error(f"Model load failed for '{model_id}': {exc}")
            print(f"[vimin] ERROR: model load failed for '{model_id}': {exc}", flush=True)
            return False

    async def _run_task_with_model(self, task: Task, model_id: str) -> None:
        """Ensure the requested model is loaded, then execute the task."""
        if not self.orchestrator.is_generative_model_loaded() or self._loaded_model_id != model_id:
            loaded = await self._load_model_async(model_id)
            if not loaded:
                logger.error(f"Cannot execute task {task.id}: model '{model_id}' failed to load")
                print(f"[vimin] ERROR: cannot run task {task.id} — model failed to load", flush=True)
                # Report failure so the center doesn't wait the full timeout
                await self._report_task_completion({
                    "task_id": task.id,
                    "task_type": "text_generation",
                    "success": False,
                    "result": "",
                    "error": f"Model '{model_id}' failed to load",
                    "execution_time_ms": 0,
                    "execution_target": "local",
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                })
                return
        await self.execute_task(task)

    async def _run_continuous_task(
        self,
        task_dict: Dict[str, Any],
        data_policy: Dict[str, Any],
    ) -> None:
        """Execute a continuous task, looping until an end condition is met.

        End conditions (checked after each iteration, any one stops the loop):
          • timeout_s      — wall-clock seconds since the task started
          • output_pattern — regex matched against the iteration's output text
          • stop_task      — explicit command from the center node

        While the agent is offline (center node unreachable), the loop
        continues autonomously.  When connectivity is restored the center
        node can send a stop_task command to halt it.
        """
        task_id = task_dict.get('id') or task_dict.get('task_id') or str(uuid.uuid4())
        end_conditions = task_dict.get('end_conditions') or {}
        timeout_s: Optional[float] = end_conditions.get('timeout_s')
        output_pattern: Optional[str] = end_conditions.get('output_pattern')
        interval_s: float = float(end_conditions.get('interval_s', 1.0))
        compiled_pattern = re.compile(output_pattern) if output_pattern else None

        stop_event = asyncio.Event()
        self._continuous_tasks[task_id] = stop_event
        start_wall = time.monotonic()
        iteration = 0

        logger.info(
            f"Continuous task {task_id} started "
            f"(timeout={timeout_s}s, pattern={output_pattern!r}, interval={interval_s}s)"
        )

        try:
            while not stop_event.is_set():
                # ── End condition: timeout ────────────────────────────────
                if timeout_s and (time.monotonic() - start_wall) >= timeout_s:
                    logger.info(f"Continuous task {task_id}: timeout reached ({timeout_s}s)")
                    break

                iter_task = Task(
                    type=TaskType[task_dict.get('type', 'TEXT_GENERATION').upper()],
                    data=task_dict.get('data', ''),
                    complexity=TaskComplexity[task_dict.get('complexity', 'low').upper()],
                    id=f"{task_id}_i{iteration}",
                    metadata=task_dict.get('metadata', {}),
                )

                result = await self.execute_task(iter_task)
                output = result.get('result', '') or ''

                # ── End condition: output pattern match ───────────────────
                if compiled_pattern and compiled_pattern.search(output):
                    logger.info(
                        f"Continuous task {task_id}: output matched pattern "
                        f"{output_pattern!r} on iteration {iteration}"
                    )
                    break

                iteration += 1

                # Wait for the inter-iteration interval, but wake immediately
                # if a stop_task command arrives.
                try:
                    await asyncio.wait_for(
                        asyncio.shield(stop_event.wait()), timeout=interval_s
                    )
                    # stop_event fired during sleep → exit on next loop check
                except asyncio.TimeoutError:
                    pass

        finally:
            self._continuous_tasks.pop(task_id, None)
            elapsed_ms = (time.monotonic() - start_wall) * 1000
            summary = {
                'task_id': task_id,
                'task_type': task_dict.get('type', 'text_generation'),
                'success': True,
                'result': f'Completed {iteration} iteration(s) in {elapsed_ms / 1000:.1f}s',
                'execution_time_ms': elapsed_ms,
                'execution_target': 'continuous_local',
                'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
                'continuous_iterations': iteration,
            }
            # Apply data policy before reporting
            blocked = data_policy.get('blocked_fields', [])
            for field in blocked:
                if field in summary:
                    summary[field] = '[BLOCKED_BY_POLICY]'
            self.task_history.append(summary)
            await self._report_task_completion(summary)
            logger.info(
                f"Continuous task {task_id} finished: {iteration} iteration(s), "
                f"{elapsed_ms:.0f} ms total"
            )

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """
        Execute a task locally and report results to the center node.

        When the task metadata contains ``stream: true`` and a generative model
        is loaded, tokens are streamed to the center node in small batches as
        they are produced (via POST /api/agents/task-stream).  This gives the
        center node — and any connected dashboard WebSocket clients — live
        token-by-token visibility into long generation tasks.

        Falls back to standard blocking execution for ONNX / encoder models.
        """
        start_time = time.time()
        task_id = task.id if hasattr(task, "id") else f"task_{int(start_time)}"
        task_type_str = (
            task.type.value
            if hasattr(task, "type") and hasattr(task.type, "value")
            else str(getattr(task, "type", "unknown"))
        )
        print(f"[vimin] Task received: {task_id} — running inference ...", flush=True)

        # If the model is still loading (e.g. a queued task arrives on reconnect
        # before the background model-load finishes), wait up to 30 s for it to
        # be ready before attempting inference.
        task_type = getattr(task, "type", None)
        needs_generative = task_type not in (TaskType.SPEECH_TO_TEXT,)
        if needs_generative and not self._model_ready.is_set():
            try:
                await asyncio.wait_for(asyncio.shield(self._model_ready.wait()), timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning(f"Task {task_id}: model not ready after 30 s, proceeding anyway")

        try:
            # ----------------------------------------------------------------
            # SPEECH_TO_TEXT — route to WhisperBackend (mlx-whisper on Apple
            # Silicon).  task.data should be a local audio file path.  The
            # model_id field (or metadata.model_id) selects the checkpoint;
            # defaults to whisper-base for lowest latency.
            # ----------------------------------------------------------------
            if getattr(task, "type", None) == TaskType.SPEECH_TO_TEXT:
                model_id = (
                    getattr(task, "model_id", None)
                    or (task.metadata.get("model_id") if hasattr(task, "metadata") else None)
                    or "openai/whisper-base"
                )
                audio_path = task.data if isinstance(task.data, str) else str(task.data)
                language = task.metadata.get("language") if hasattr(task, "metadata") else None

                try:
                    from vimin_core.core.backends.whisper_backend import WhisperBackend
                    from vimin_core.core.backends import ModelDescriptor
                except ImportError as exc:
                    raise RuntimeError(
                        f"WhisperBackend not available — install mlx-whisper: "
                        f"pip install 'vimin-core[whisper]'"
                    ) from exc

                async with self._inference_lock:
                    if self._whisper_backend is None:
                        self._whisper_backend = WhisperBackend()
                    if self._whisper_model_id != model_id:
                        descriptor = ModelDescriptor(model_id=model_id)
                        loaded = await asyncio.to_thread(self._whisper_backend.load, descriptor)
                        if not loaded:
                            raise RuntimeError(f"Whisper model '{model_id}' failed to load")
                        self._whisper_model_id = model_id
                        print(f"[vimin] Whisper model ready: {model_id}", flush=True)

                    result_dict = await asyncio.to_thread(
                        self._whisper_backend.transcribe, audio_path, language
                    )

                output = result_dict.get("text", "").strip()
                success = True
                execution_target = "local_whisper"
                execution_time = time.time() - start_time
                print(f"[vimin] Inference complete: {task_id} ({execution_time:.1f}s)", flush=True)
                if output:
                    print(f"[vimin] Output:\n{output}", flush=True)
                save_local = task.metadata.get("save_local", False) if hasattr(task, "metadata") else False
                result_for_center, saved_path = self._handle_save_local(
                    task_id, task_type_str, output, save_local
                )
                task_record = {
                    "task_id": task_id,
                    "task_type": task_type_str,
                    "success": success,
                    "result": result_for_center,
                    "execution_time_ms": execution_time * 1000,
                    "execution_target": execution_target,
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                }
                if saved_path:
                    task_record["output_path"] = saved_path
                self.task_history.append(task_record)
                await self._report_task_completion(task_record)
                return {"success": True, "result": output, "execution_time_ms": execution_time * 1000}

            if self.orchestrator:
                stream_requested = (
                    task.metadata.get("stream", False)
                    if hasattr(task, "metadata") else False
                )
                use_streaming = (
                    stream_requested
                    and self.orchestrator.is_generative_model_loaded()
                )

                if use_streaming:
                    # --------------------------------------------------------
                    # Streaming path: yield tokens to center node as they arrive.
                    # The generator runs in a thread-pool executor so each token
                    # computation (CPU/GPU-bound) does not block the event loop.
                    # Tokens are passed back via an asyncio.Queue so heartbeats
                    # and command polling can interleave between batches.
                    # The inference lock is held for the full duration so only
                    # one backend call is in flight at a time.
                    # --------------------------------------------------------
                    output_chunks: list[str] = []
                    buffer: list[str] = []
                    _BATCH_TOKENS = 8  # POST every N tokens for low latency

                    _loop = asyncio.get_running_loop()
                    _token_queue: asyncio.Queue = asyncio.Queue()

                    def _stream_in_thread():
                        try:
                            for _tok in self.orchestrator.stream_execute_task(task):
                                _loop.call_soon_threadsafe(_token_queue.put_nowait, _tok)
                        except Exception as _exc:
                            _loop.call_soon_threadsafe(_token_queue.put_nowait, _exc)
                        finally:
                            _loop.call_soon_threadsafe(_token_queue.put_nowait, None)

                    async with self._inference_lock:
                        _thread_future = _loop.run_in_executor(None, _stream_in_thread)
                        while True:
                            item = await _token_queue.get()
                            if item is None:
                                break
                            if isinstance(item, BaseException):
                                await _thread_future
                                raise item
                            buffer.append(item)
                            output_chunks.append(item)
                            if len(buffer) >= _BATCH_TOKENS:
                                await self._report_partial_result(
                                    task_id, "".join(buffer), len("".join(output_chunks))
                                )
                                buffer = []
                        await self._report_partial_result(
                            task_id,
                            "".join(buffer),
                            len("".join(output_chunks)),
                            final=True,
                        )
                        await _thread_future

                    output = "".join(output_chunks)
                    success = True
                    execution_target = "local_generative"

                else:
                    # --------------------------------------------------------
                    # Standard blocking path (ONNX encoder or no stream flag)
                    # Run in a thread pool so the event loop stays alive for
                    # heartbeats and command polling during long inference.
                    # The lock ensures only one thread uses the backend at a
                    # time (MLX / Metal are not concurrent-thread-safe).
                    # --------------------------------------------------------
                    async with self._inference_lock:
                        result = await asyncio.to_thread(self.orchestrator.execute_task, task)
                    success = result.success if hasattr(result, "success") else True
                    output = (
                        result.result
                        if hasattr(result, "result") and result.result is not None
                        else result.output if hasattr(result, "output") else "Task completed"
                    )
                    raw_target = (
                        result.execution_target if hasattr(result, "execution_target") else "local"
                    )
                    execution_target = (
                        raw_target.value if hasattr(raw_target, "value") else str(raw_target)
                    )

            else:
                # Demo / fallback when no orchestrator is wired
                await asyncio.sleep(0.05)
                success = True
                output = f"Demo: processed '{getattr(task, 'data', '')}'"
                execution_target = "demo"

            execution_time = time.time() - start_time
            print(f"[vimin] Inference complete: {task_id} ({execution_time:.1f}s)", flush=True)

            # Always print the output on the edge node so it's visible in the agent log
            output_str = (output[:32768] if isinstance(output, str) else "") if output else ""
            if output_str:
                print(f"[vimin] Output:\n{output_str}", flush=True)

            save_local = task.metadata.get("save_local", False) if hasattr(task, "metadata") else False
            result_for_center, saved_path = self._handle_save_local(
                task_id, task_type_str, output_str, save_local
            )

            task_record = {
                "task_id": task_id,
                "task_type": task_type_str,
                "success": success,
                "result": result_for_center,
                "execution_time_ms": execution_time * 1000,
                "execution_target": execution_target,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
            if saved_path:
                task_record["output_path"] = saved_path
            self.task_history.append(task_record)
            await self._report_task_completion(task_record)

            return {
                "success": success,
                "result": output,
                "execution_time_ms": execution_time * 1000,
                "execution_target": execution_target,
            }

        except Exception as exc:
            logger.error(f"Task {task_id} execution failed: {exc}")
            print(f"[vimin] ERROR: task {task_id} failed — {exc}", flush=True)
            execution_time = time.time() - start_time
            task_record = {
                "task_id": task_id,
                "task_type": task_type_str,
                "success": False,
                "result": "",
                "error": str(exc),
                "execution_time_ms": execution_time * 1000,
                "execution_target": "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }
            self.task_history.append(task_record)
            await self._report_task_completion(task_record)
            return {
                "success": False,
                "error": str(exc),
                "execution_time_ms": execution_time * 1000,
            }

    def _handle_save_local(
        self,
        task_id: str,
        task_type: str,
        output: str,
        save_local: bool,
    ) -> tuple:
        """
        If save_local is True, write output to ~/.vimin/outputs/ and return a
        lightweight reference string for the center node instead of the full text.
        Returns (result_for_center, saved_path_or_None).
        """
        if not save_local or not output:
            return output, None
        output_dir = os.path.join(os.path.expanduser("~"), ".vimin", "outputs")
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_type = task_type.lower().replace("_", "-")
        filename = f"{ts}_{safe_type}_{task_id[:12]}.txt"
        path = os.path.join(output_dir, filename)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(output)
            print(f"[vimin] Output saved: {path}", flush=True)
            return f"[saved_locally] {path}", path
        except Exception as exc:
            logger.warning(f"Could not save output locally: {exc}")
            return output, None

    async def _report_partial_result(
        self,
        task_id: str,
        token_chunk: str,
        chars_so_far: int,
        final: bool = False,
    ) -> None:
        """
        POST a batch of streaming tokens to the center node.

        The center node relays these to all connected WebSocket dashboard
        clients so the UI can display tokens as they arrive.
        """
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            if self._agent_secret:
                headers["X-Vimin-Agent-Secret"] = self._agent_secret
            payload = {
                "agent_id": self.agent_id,
                "task_id": task_id,
                "token_chunk": token_chunk,
                "chars_so_far": chars_so_far,
                "final": final,
            }
            async with self.session.post(
                f"{self.center_node_url}/api/agents/task-stream",
                json=payload,
                headers=headers,
            ) as response:
                if response.status not in (200, 202):
                    logger.debug(
                        f"task-stream POST returned {response.status} for task {task_id}"
                    )
        except Exception as exc:
            # Never let streaming reporting block or crash inference
            logger.debug(f"_report_partial_result failed (non-fatal): {exc}")
    
    async def _report_task_completion(self, task_record: Dict[str, Any]):
        """Report task completion to center node with privacy redaction.
        When the center node is unreachable the record is written to the local
        offline buffer so it can be replayed when connectivity is restored."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        if self._agent_secret:
            headers["X-Vimin-Agent-Secret"] = self._agent_secret

        # Apply privacy filters
        report_data = task_record.copy()
        if self.privacy_mode:
            for sensitive in ["raw_text", "input_data", "output"]:
                if sensitive in report_data:
                    report_data[sensitive] = "[REDACTED_BY_CLIENT]"
            if report_data.get('task_type') != TaskType.PII_MASKING.value:
                report_data['result'] = "[REDACTED_BY_POLICY]"

        payload = {"agent_id": self.agent_id, "task_record": report_data}

        if not self._connected:
            self._buffer_locally("/api/agents/task-completion", payload)
            return

        # Use urllib (a fresh TCP connection each time) rather than the shared
        # aiohttp session.  After long inference runs the aiohttp session's
        # internal keep-alive connections go stale; a subsequent POST silently
        # times out, the result gets buffered locally, and the center never
        # receives it.  urllib avoids shared session state entirely.
        url = f"{self.center_node_url}/api/agents/task-completion"
        try:
            await asyncio.to_thread(self._post_completion_sync, url, payload, headers)
            logger.info(
                f"Reported task completion: task={task_record.get('task_id') or task_record.get('id')} "
                f"success={task_record.get('success', True)}"
            )
        except Exception as e:
            logger.warning(f"Failed to report task completion (buffering locally): {e}")
            self._buffer_locally("/api/agents/task-completion", payload)

    def _post_completion_sync(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> None:
        """Synchronous urllib POST — called via asyncio.to_thread so it doesn't block the event loop.
        Uses a fresh TCP connection each time to avoid stale aiohttp keep-alive issues."""
        import urllib.request as _urlreq
        data = json.dumps(payload).encode()
        req = _urlreq.Request(
            url,
            data=data,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        with _urlreq.urlopen(req, timeout=15) as resp:
            if resp.status not in (200, 201, 202):
                raise RuntimeError(f"Task completion POST returned {resp.status}")

    def _buffer_locally(self, path: str, payload: Dict[str, Any]) -> None:
        """Append a failed POST to the local offline buffer file (NDJSON)."""
        entry = {"path": path, "payload": payload, "buffered_at": time.time()}
        try:
            with open(self._offline_buffer_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Could not write to offline buffer: {e}")

    async def _flush_offline_buffer(self) -> None:
        """Replay all locally-buffered events to the center node.
        Called automatically when the center node becomes reachable again."""
        if not os.path.exists(self._offline_buffer_path):
            return
        try:
            with open(self._offline_buffer_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            entries = [json.loads(l) for l in lines]
            os.unlink(self._offline_buffer_path)
        except Exception as e:
            logger.warning(f"Could not read offline buffer: {e}")
            return

        if not entries:
            return

        logger.info(f"Flushing {len(entries)} buffered event(s) to center node")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        if self._agent_secret:
            headers["X-Vimin-Agent-Secret"] = self._agent_secret
        failed: list = []
        for entry in entries:
            try:
                url = f"{self.center_node_url}{entry['path']}"
                await asyncio.to_thread(self._post_completion_sync, url, entry["payload"], headers)
            except Exception:
                failed.append(entry)

        # Re-buffer anything that still failed
        if failed:
            try:
                with open(self._offline_buffer_path, "w") as f:
                    for entry in failed:
                        f.write(json.dumps(entry) + "\n")
            except Exception:
                pass
        else:
            logger.info("Offline buffer flushed successfully")


# CLI interface for running the user agent
async def main():
    """Main entry point for user agent"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="vimin User Agent",
        epilog=(
            "Env vars: VIMIN_CENTER_URL (overrides --center-node), "
            "ORCHESTRATOR_API_KEY, ORCHESTRATOR_PRIVACY_MODE"
        ),
    )
    parser.add_argument(
        "--center-node",
        default=os.environ.get("VIMIN_CENTER_URL", "http://localhost:8080"),
        help="Center node URL (default: $VIMIN_CENTER_URL or http://localhost:8080)",
    )
    parser.add_argument("--agent-id", help="Agent ID (auto-generated if not provided)")
    parser.add_argument("--api-key", help="API Key for authentication")
    parser.add_argument(
        "--model",
        default=os.environ.get("VIMIN_DEFAULT_MODEL"),
        metavar="MODEL_ID",
        help=(
            "HuggingFace model ID to pre-load on startup, e.g. "
            "meta-llama/Llama-3.2-1B-Instruct  (default: $VIMIN_DEFAULT_MODEL)"
        ),
    )
    parser.add_argument("--privacy", action="store_true", help="Enable privacy mode (redact sensitive data)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    tls_group = parser.add_argument_group("TLS / HTTPS")
    tls_group.add_argument(
        "--tls-ca", metavar="CA.pem",
        help="Path to CA certificate for verifying a self-signed center node cert",
    )
    tls_group.add_argument(
        "--no-tls-verify", action="store_true",
        help="Disable TLS certificate verification (development only)",
    )
    parser.add_argument(
        "--fleet-token",
        default=os.environ.get("VIMIN_FLEET_TOKEN"),
        help="Fleet enrollment token (default: $VIMIN_FLEET_TOKEN)",
    )

    args = parser.parse_args()

    from vimin_core.utils.log_config import configure_logging
    configure_logging(logging.DEBUG if args.debug else logging.INFO)

    agent = UserAgent(
        center_node_url=args.center_node,
        agent_id=args.agent_id,
        api_key=args.api_key,
        privacy_mode=args.privacy,
        tls_ca=args.tls_ca,
        tls_verify=not args.no_tls_verify,
        fleet_token=args.fleet_token,
    )
    
    import signal as _signal

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    # SIGTERM (sent by `vimin-core stop-agent`) and Ctrl+C both trigger
    # graceful shutdown so the goodbye heartbeat is always delivered.
    loop.add_signal_handler(_signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(_signal.SIGINT, stop_event.set)

    await agent.start()
    print(f"User Agent running. ID: {agent.agent_id}")
    if args.model:
        print(f"Pre-loading model: {args.model}")
        ok = await agent._load_model_async(args.model)
        if ok:
            print(f"Model ready: {args.model}")
        else:
            print(f"Warning: model pre-load failed for {args.model}")
    print("Press Ctrl+C to stop...")

    await stop_event.wait()
    print("\nShutting down...")
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())

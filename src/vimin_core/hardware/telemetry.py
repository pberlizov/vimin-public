"""
Hardware Telemetry Module

Monitors system resources including RAM, CPU load, battery status,
thermal state, and NPU availability detection.
Includes a background TelemetryCollector thread for continuous monitoring.
"""

import psutil # type: ignore
import platform
import subprocess
import time
import logging
import threading
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from vimin_core.hardware.scanner import HardwareScanner, HardwareVendor  # type: ignore

# Configure logger for this module
logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Data class to hold system telemetry metrics"""
    available_ram_gb: float
    cpu_load_percent: float
    battery_percent: Optional[float]
    is_plugged_in: bool
    npu_available: bool
    soc_name: str
    npu_utilization_percent: float = 0.0
    thermal_state_celsius: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class HardwareTelemetry:
    """
    Hardware telemetry monitoring class that provides real-time system metrics
    for routing decisions in the AI orchestration system.
    """
    
    def __init__(self):
        """Initialize the telemetry monitor"""
        self.system = platform.system()
        self.machine = platform.machine()
        self.scanner = HardwareScanner()
        self._thermal_cache: Optional[float] = None
        self._thermal_cache_time: float = 0.0
        
    def get_available_ram_gb(self) -> float:
        """
        Get available RAM in gigabytes
        
        Returns:
            float: Available RAM in GB
        """
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)  # Convert bytes to GB
    
    def get_cpu_load_percent(self) -> float:
        """
        Get current CPU load percentage
        
        Returns:
            float: CPU load percentage (0-100)
        """
        return psutil.cpu_percent(interval=None)
    
    def get_battery_status(self) -> tuple[Optional[float], bool]:
        """
        Get battery status if available
        
        Returns:
            tuple: (battery_percent, is_plugged_in)
                   battery_percent is None if no battery is detected
        """
        try:
            battery = psutil.sensors_battery()
            if battery is None:
                return None, False
            return float(battery.percent), battery.power_plugged
        except (AttributeError, OSError):
            # Battery sensors not available on this system
            return None, False
    
    def get_npu_availability(self) -> bool:
        """
        Check for NPU availability using the unified HardwareScanner
        
        Returns:
            bool: True if NPU is available (vendor is not GENERIC/UNKNOWN)
        """
        cap = self.scanner.get_capability()
        return cap.vendor not in [HardwareVendor.GENERIC, HardwareVendor.UNKNOWN]
    
    def get_thermal_state(self) -> Optional[float]:
        """
        Get device temperature in Celsius.
        Uses cached value if polled within the last 2 seconds (thermal reads are expensive).
        
        Returns:
            Optional[float]: Temperature in Celsius, or None if unavailable
        """
        now = time.time()
        if self._thermal_cache is not None and (now - self._thermal_cache_time) < 2.0:
            return self._thermal_cache
        
        temp = None
        
        if self.system == "Darwin":
            temp = self._read_macos_temperature()
        elif self.system == "Linux":
            temp = self._read_linux_temperature()
        elif self.system == "Windows":
            temp = self._read_windows_temperature()
        
        if temp is not None:
            self._thermal_cache = temp
            self._thermal_cache_time = now
        
        return temp

    def get_npu_utilization(self) -> float:
        """
        Get current NPU utilization percentage.
        
        Returns:
            float: NPU utilization (0-100)
        """
        if self.system == "Darwin":
            return self._read_macos_npu_utilization()
        elif self.system == "Linux":
            return self._read_linux_npu_utilization()
        elif self.system == "Windows":
            return self._read_windows_npu_utilization()
        return 0.0

    def _read_macos_npu_utilization(self) -> float:
        """
        Read Apple Neural Engine (ANE) utilization using powermetrics.
        Requires sudo for real metrics.
        """
        try:
            # We use a single sample snapshot to keep it lightweight
            # powermetrics -i 1 -n 1 --samplers cpu_power --format json
            cmd = ["sudo", "-n", "powermetrics", "-i", "1", "-n", "1", "--samplers", "cpu_power", "--format", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                # The output sometimes contains a header before the JSON
                output: str = result.stdout
                json_start = output.find('{')
                if json_start != -1:
                    data = json.loads(output[json_start:])
                    # M1/M2/M3/M4 structures vary, but 'processor' -> 'ane_energy' or 'ane_utilization' is common
                    processor = data.get('processor', {})
                    
                    # Try direct utilization first (newest macOS versions)
                    if 'ane_utilization' in processor:
                        return float(processor['ane_utilization'])
                    
                    # Fallback: estimate from energy if available
                    if 'ane_energy' in processor:
                        # energy is in mWatts, let's normalize roughly (peak is ~1000-2000mW)
                        energy = float(processor['ane_energy'])
                        return min(100.0, (energy / 1500.0) * 100.0)
                        
                    # Some versions put it under 'gpu' or other keys
                    for key, val in data.items():
                        if isinstance(val, dict) and 'ane_utilization' in val:
                            return float(val['ane_utilization'])

        except Exception as e:
            logger.debug(f"Powermetrics parsing failed: {e}")
            
        return 0.0

    def _read_linux_npu_utilization(self) -> float:
        """Read NPU utilization on Linux (Intel/NVIDIA)"""
        try:
            # Check for NVIDIA first
            result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], 
                                   capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                return float(result.stdout.strip())
            
            # Check for Intel NPU via /sys/class/accel (if Linux 6.3+)
            accel_dir = "/sys/class/accel/accel0/device/utilization"
            if os.path.exists(accel_dir):
                with open(accel_dir, "r") as f:
                    return float(f.read().strip())
        except Exception:
            pass
        return 0.0

    def _read_windows_npu_utilization(self) -> float:
        """Read NPU utilization on Windows via Performance Counters or wmic"""
        # Placeholder for complex Windows WMI query
        return 0.0

    def _read_windows_temperature(self) -> Optional[float]:
        """Read CPU temperature on Windows.
        Tries OpenHardwareMonitor WMI namespace first; falls back to a CPU%
        proxy so the routing guards always have a usable value."""
        try:
            import wmi  # type: ignore
            w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
            sensors = w.Sensor()
            for sensor in sensors:
                if sensor.SensorType == "Temperature" and "CPU" in sensor.Name:
                    return float(sensor.Value)
        except Exception:
            pass
        # psutil-based proxy: idle ~35°C, full load ~85°C
        try:
            cpu = psutil.cpu_percent(interval=None)
            return 35.0 + (cpu / 100.0) * 50.0
        except Exception:
            return None

    def _read_macos_temperature(self) -> Optional[float]:
        """Read CPU die temperature on macOS via powermetrics or a simulated fallback"""
        try:
            # Try reading from IOKit via a lightweight subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                # thermal_level is 0-127; map roughly to celsius
                level = int(result.stdout.strip())
                # Approximate: level 0 = ~35C, level 127 = ~105C
                return 35.0 + (level / 127.0) * 70.0
        except Exception:
            pass
        
        # Fallback: use psutil cpu_percent as a rough proxy
        try:
            cpu = psutil.cpu_percent(interval=None)
            return 32.0 + (cpu / 100.0) * 45.0
        except Exception:
            return None

    def _read_linux_temperature(self) -> Optional[float]:
        """Read CPU temperature on Linux via psutil sensors"""
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return None
            for name in ['coretemp', 'cpu_thermal', 'k10temp', 'zenpower']:
                if name in temps:
                    readings = temps[name]
                    if readings:
                        return readings[0].current
            first_key = next(iter(temps))
            if temps[first_key]:
                return temps[first_key][0].current
        except Exception:
            return None
        return None

    def get_system_metrics(self) -> SystemMetrics:
        """
        Get comprehensive system metrics
        
        Returns:
            SystemMetrics: Current system telemetry data
        """
        battery_percent, is_plugged_in = self.get_battery_status()
        
        return SystemMetrics(
            available_ram_gb=self.get_available_ram_gb(),
            cpu_load_percent=self.get_cpu_load_percent(),
            battery_percent=battery_percent,
            is_plugged_in=is_plugged_in,
            npu_available=self.get_npu_availability(),
            npu_utilization_percent=self.get_npu_utilization(),
            soc_name=self.scanner.get_capability().soc_name,
            thermal_state_celsius=self.get_thermal_state(),
            timestamp=time.time()
        )
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """
        Get a dictionary summary of current telemetry data
        
        Returns:
            Dict[str, Any]: Summary of system metrics
        """
        metrics = self.get_system_metrics()
        return {
            "available_ram_gb": float("{:.2f}".format(metrics.available_ram_gb)),
            "cpu_load_percent": float("{:.1f}".format(metrics.cpu_load_percent)),
            "npu_utilization_percent": float("{:.1f}".format(metrics.npu_utilization_percent)),
            "battery_percent": metrics.battery_percent,
            "is_plugged_in": metrics.is_plugged_in,
            "npu_available": metrics.npu_available,
            "thermal_state_celsius": float("{:.1f}".format(metrics.thermal_state_celsius)) if metrics.thermal_state_celsius is not None else None,
            "thermal_unit": "Celsius",
            "platform": f"{self.system} {self.machine}",
            "soc": self.scanner.get_capability().soc_name,
            "timestamp": metrics.timestamp
        }


class TelemetryCollector:
    """
    Background daemon thread that continuously polls system state.
    Provides lock-free reads of the latest metrics snapshot.
    """
    
    def __init__(self, poll_interval: float = 1.0):
        """
        Args:
            poll_interval: Seconds between polls (default 1.0)
        """
        self._telemetry = HardwareTelemetry()
        self._poll_interval = poll_interval
        self._latest: Optional[SystemMetrics] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
    
    def start(self) -> bool:
        """Start the background collector thread"""
        if self._thread is not None and self._thread.is_alive():
            return False # Already running
        self._stop_event.clear()
        thread = threading.Thread(target=self._poll_loop, daemon=True, name="TelemetryCollector")
        self._thread = thread
        thread.start()
        logger.info(f"TelemetryCollector started (interval={self._poll_interval}s)")
        return True
    
    def stop(self):
        """Stop the background polling thread"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        logger.info("TelemetryCollector stopped")
    
    def _poll_loop(self):
        """Internal polling loop running on the daemon thread"""
        while not self._stop_event.is_set():
            try:
                snapshot = self._telemetry.get_system_metrics()
                with self._lock:
                    self._latest = snapshot
            except Exception as e:
                logger.warning(f"Telemetry poll error: {e}")
            self._stop_event.wait(timeout=self._poll_interval)
    
    def get_latest(self) -> SystemMetrics:
        """
        Get the most recent metrics snapshot (thread-safe).
        Falls back to a live read if no snapshot is available yet.
        """
        with self._lock:
            latest = self._latest
            if latest is not None:
                return latest
        # Fallback: synchronous read
        return self._telemetry.get_system_metrics()
    
    @property
    def telemetry(self) -> HardwareTelemetry:
        """Access the underlying HardwareTelemetry instance"""
        return self._telemetry
    
    @property
    def is_running(self) -> bool:
        """Thread-safe status check"""
        return self._thread is not None and self._thread.is_alive()


# Example usage and testing
if __name__ == "__main__":
    telemetry = HardwareTelemetry()
    print("=== Hardware Telemetry Test ===")
    print(telemetry.get_telemetry_summary())
    
    print("\n=== TelemetryCollector Test ===")
    collector = TelemetryCollector(poll_interval=0.5)
    collector.start()
    time.sleep(2)
    latest = collector.get_latest()
    print(f"Latest: RAM={latest.available_ram_gb:.1f}GB, CPU={latest.cpu_load_percent:.1f}%, Thermal={latest.thermal_state_celsius}")
    collector.stop()
    print("Collector stopped.")

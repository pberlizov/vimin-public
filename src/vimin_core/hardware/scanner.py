"""
Hardware Scanner Module

Provides vendor-agnostic detection of SoC capabilities and NPU hardware.
Abstractions for Apple Silicon, Intel Ultra (OpenVINO), and Qualcomm (QNN).
"""

import platform
import subprocess
import json
import logging
import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, cast

logger = logging.getLogger(__name__)

class HardwareVendor(Enum):
    APPLE = "Apple"
    INTEL = "Intel"
    QUALCOMM = "Qualcomm"
    NVIDIA = "NVIDIA"
    GENERIC = "Generic"
    UNKNOWN = "Unknown"

@dataclass
class SoCCapability:
    """Hardware capability profile"""
    vendor: HardwareVendor
    soc_name: str
    npu_name: str
    primary_provider: str
    supports_fp16: bool = True
    is_arm: bool = False

class HardwareScanner:
    """
    Unified hardware detection engine for Cross-SoC support.
    """
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self._capability: Optional[SoCCapability] = None

    def get_capability(self) -> SoCCapability:
        """Get the detected hardware capability profile"""
        if self._capability is not None:
            return cast(SoCCapability, self._capability)
        
        cap: SoCCapability
        if self.system == "Darwin":
            cap = self._scan_apple()
        elif self.system == "Windows":
            cap = self._scan_windows()
        elif self.system == "Linux":
            cap = self._scan_linux()
        else:
            cap = SoCCapability(
                vendor=HardwareVendor.UNKNOWN,
                soc_name="Generic Platform",
                npu_name="None",
                primary_provider="CPUExecutionProvider",
                is_arm="arm" in self.machine.lower()
            )
        
        self._capability = cap
        return cap

    def _scan_apple(self) -> SoCCapability:
        """Scan for Apple Silicon NPU"""
        soc_name = "Apple Silicon"
        is_arm = self.machine.lower() in ["arm64", "arm"]
        
        if is_arm:
            try:
                # Try to get more specific model info
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    soc_name = result.stdout.strip()
            except Exception:
                pass
                
            return SoCCapability(
                vendor=HardwareVendor.APPLE,
                soc_name=soc_name,
                npu_name="Apple Neural Engine",
                primary_provider="CoreMLExecutionProvider",
                is_arm=True
            )
        
        return SoCCapability(
            vendor=HardwareVendor.GENERIC,
            soc_name="Intel Mac",
            npu_name="None",
            primary_provider="CPUExecutionProvider",
            is_arm=False
        )

    def _scan_windows(self) -> SoCCapability:
        """Scan for Windows NPUs and GPUs"""
        try:
            # 1. Check for NVIDIA GPU first (high priority for most)
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                                       capture_output=True, text=True, timeout=1)
                if result.returncode == 0:
                    gpu_name = result.stdout.strip()
                    return SoCCapability(
                        vendor=HardwareVendor.NVIDIA,
                        soc_name="Windows PC",
                        npu_name=gpu_name,
                        primary_provider="CUDAExecutionProvider"
                    )
            except Exception:
                pass

            # 2. Check for DirectML (any GPU)
            # This is a safe fallback for Windows
            
            brand = self._get_windows_cpu_brand().upper()
            
            if "SNAPDRAGON" in brand or "QUALCOMM" in brand:
                return SoCCapability(
                    vendor=HardwareVendor.QUALCOMM,
                    soc_name=brand or "Snapdragon Processor",
                    npu_name="Hexagon NPU",
                    primary_provider="QNNExecutionProvider",
                    is_arm=True
                )
            
            if "ULTRA" in brand and "INTEL" in brand:
                return SoCCapability(
                    vendor=HardwareVendor.INTEL,
                    soc_name=brand or "Intel Core Ultra",
                    npu_name="Intel AI Boost",
                    primary_provider="OpenVINOExecutionProvider",
                    supports_fp16=True
                )
        except Exception as e:
            logger.debug(f"Windows specific scan failed: {e}")

        return SoCCapability(
            vendor=HardwareVendor.GENERIC,
            soc_name="Windows PC",
            npu_name="DirectML (GPU/CPU)",
            primary_provider="DMLExecutionProvider"
        )

    def _scan_linux(self) -> SoCCapability:
        """Scan for Linux NPUs (Intel, NVIDIA)"""
        try:
            # 1. Check for NVIDIA
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                                       capture_output=True, text=True, timeout=1)
                if result.returncode == 0:
                    return SoCCapability(
                        vendor=HardwareVendor.NVIDIA,
                        soc_name="Linux Machine",
                        npu_name=result.stdout.strip(),
                        primary_provider="CUDAExecutionProvider"
                    )
            except Exception:
                pass

            # 2. Check for Intel NPU viaAccel
            if os.path.exists("/dev/accel") or os.path.exists("/sys/class/accel"):
                return SoCCapability(
                    vendor=HardwareVendor.INTEL,
                    soc_name="Intel Core Ultra (Detected)",
                    npu_name="Intel AI Boost",
                    primary_provider="OpenVINOExecutionProvider"
                )

            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read().upper()
                
            if "INTEL" in cpuinfo and ("ULTRA" in cpuinfo or "CORE ULTRA" in cpuinfo):
                return SoCCapability(
                    vendor=HardwareVendor.INTEL,
                    soc_name="Intel Core Ultra (Heuristic)",
                    npu_name="Intel AI Boost",
                    primary_provider="OpenVINOExecutionProvider"
                )
        except Exception:
            pass

        return SoCCapability(
            vendor=HardwareVendor.GENERIC,
            soc_name="Linux Machine",
            npu_name="Generic",
            primary_provider="CPUExecutionProvider",
            is_arm="arm" in self.machine.lower()
        )

    def _get_windows_cpu_brand(self) -> str:
        """Helper to get CPU brand on Windows"""
        try:
            import wmi # type: ignore
            c = wmi.WMI()
            for processor in c.Win32_Processor():
                return str(processor.Name)
        except (ImportError, Exception):
            # Fallback to shell if wmi is missing or fails
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        return lines[1].strip()
            except Exception:
                pass
        return "Windows Generic x86_64"

    def get_provider_options(self, provider: str) -> Dict[str, Any]:
        """Get optimized options for a specific provider based on hardware"""
        cap = self.get_capability()
        
        if provider == 'CoreMLExecutionProvider':
            return {'MLComputeUnits': 'ALL'}
        
        if provider == 'OpenVINOExecutionProvider':
            return {
                'device_type': 'NPU' if cap.vendor == HardwareVendor.INTEL else 'CPU',
                'enable_npu_fast_compile': 'True',
                'cache_dir': '.cache/onnx_models'
            }
            
        if provider == 'QNNExecutionProvider':
            return {
                'backend_path': 'QnnHtp.dll', # Target Hexagon
                'htp_performance_mode': 'high_performance',
                'context_enable': 'True',
                'user_context_file_path': '.cache/onnx_models'
            }
            
        if provider == 'CUDAExecutionProvider':
            return {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested'
            }
            
        return {}

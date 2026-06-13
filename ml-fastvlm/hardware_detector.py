"""
Hardware Detection Module for LiveQA Vision Engine.

Probes all available compute resources and selects the optimal backend
for FastVLM inference. Supports:
  - NVIDIA CUDA (Windows/Linux)
  - Apple MPS / Metal (macOS)
  - Apple MLX + ANE (macOS, Apple Silicon)
  - DirectML (Windows, any DX12 GPU: AMD/Intel/NVIDIA)
  - CPU fallback (any OS, AVX2/AVX-512/NEON)

Usage:
    from hardware_detector import detect_hardware, get_device_report
    hw = detect_hardware()
    print(hw.backend)   # "cuda", "mlx", "mps", "directml", "cpu"
    print(hw.device)    # torch device string
"""

import platform
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""
    backend: str            # "cuda", "mlx", "mps", "directml", "cpu"
    device: str             # torch device string (e.g. "cuda:0", "mps", "cpu")
    device_name: str        # Human-readable name (e.g. "NVIDIA RTX 4060")
    vram_gb: Optional[float] = None
    os_name: str = ""
    cpu_name: str = ""
    cpu_cores: int = 0
    has_avx2: bool = False
    has_avx512: bool = False
    torch_dtype: str = "float32"   # Recommended dtype
    optimizations: list = field(default_factory=list)  # Available optimizations


def _detect_cpu_info() -> dict:
    """Detect CPU capabilities."""
    info = {
        "name": platform.processor() or "Unknown CPU",
        "cores": os.cpu_count() or 1,
        "has_avx2": False,
        "has_avx512": False,
    }
    
    # Check SIMD capabilities on x86
    try:
        if platform.machine() in ("x86_64", "AMD64"):
            # Try to detect AVX2/AVX-512 via cpuid
            import subprocess
            if platform.system() == "Windows":
                # Windows: use wmic or just assume modern CPU has AVX2
                info["has_avx2"] = True  # Most CPUs since 2013
            else:
                result = subprocess.run(
                    ["grep", "-c", "avx2", "/proc/cpuinfo"],
                    capture_output=True, text=True, timeout=2
                )
                info["has_avx2"] = int(result.stdout.strip()) > 0
                result = subprocess.run(
                    ["grep", "-c", "avx512", "/proc/cpuinfo"],
                    capture_output=True, text=True, timeout=2
                )
                info["has_avx512"] = int(result.stdout.strip()) > 0
        elif platform.machine() == "arm64":
            info["name"] = "Apple Silicon" if platform.system() == "Darwin" else "ARM64"
    except Exception:
        pass
    
    return info


def _check_mlx() -> Optional[dict]:
    """Check if MLX (Apple ML framework) is available."""
    try:
        import mlx.core as mx
        return {
            "available": True,
            "default_device": str(mx.default_device()),
        }
    except ImportError:
        return None


def _check_cuda() -> Optional[dict]:
    """Check if NVIDIA CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "vram_gb": round(props.total_mem / 1e9, 1),
                "compute_capability": f"{props.major}.{props.minor}",
                "is_rocm": torch.version.hip is not None,
                "multi_gpu": torch.cuda.device_count(),
            }
    except Exception:
        pass
    return None


def _check_mps() -> Optional[dict]:
    """Check if Apple MPS (Metal Performance Shaders) is available."""
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "available": True,
                "name": "Apple Silicon (Metal/MPS)",
            }
    except Exception:
        pass
    return None


def _check_directml() -> Optional[dict]:
    """Check if DirectML (Windows, any DX12 GPU) is available."""
    try:
        import torch_directml
        device = torch_directml.device()
        name = torch_directml.device_name(0)
        return {
            "available": True,
            "device": str(device),
            "name": name,
        }
    except Exception:
        pass
    return None


def detect_hardware() -> HardwareInfo:
    """
    Detect the best available hardware backend.
    
    Priority order (highest performance first):
      1. MLX (Apple ANE + GPU — sub-100ms TTFT)
      2. CUDA (NVIDIA GPU — 60-200ms TTFT)
      3. MPS (Apple Metal — 100-300ms TTFT)
      4. DirectML (Any Windows GPU — 200-400ms TTFT)
      5. CPU (fallback — 500ms-2s TTFT)
    """
    cpu_info = _detect_cpu_info()
    os_name = f"{platform.system()} {platform.release()}"
    
    # --- 1. Check MLX (Apple Silicon, best for Mac) ---
    mlx_info = _check_mlx()
    if mlx_info and mlx_info["available"]:
        return HardwareInfo(
            backend="mlx",
            device="mlx",
            device_name=f"Apple Silicon (MLX — {mlx_info['default_device']})",
            os_name=os_name,
            cpu_name=cpu_info["name"],
            cpu_cores=cpu_info["cores"],
            torch_dtype="float16",
            optimizations=["mlx_compile", "ane_offload", "unified_memory"],
        )
    
    # --- 2. Check CUDA (NVIDIA GPU) ---
    cuda_info = _check_cuda()
    if cuda_info and cuda_info["available"] and not cuda_info.get("is_rocm"):
        optimizations = ["torch_compile", "flash_attention", "cuda_graphs"]
        if cuda_info["vram_gb"] >= 4:
            optimizations.append("fp16")
        if cuda_info["vram_gb"] >= 2:
            optimizations.append("int8_quantization")
        
        return HardwareInfo(
            backend="cuda",
            device="cuda:0",
            device_name=cuda_info["name"],
            vram_gb=cuda_info["vram_gb"],
            os_name=os_name,
            cpu_name=cpu_info["name"],
            cpu_cores=cpu_info["cores"],
            torch_dtype="float16",
            optimizations=optimizations,
        )
    
    # --- 3. Check MPS (Apple Metal, fallback if no MLX) ---
    mps_info = _check_mps()
    if mps_info and mps_info["available"]:
        return HardwareInfo(
            backend="mps",
            device="mps",
            device_name="Apple Silicon (MPS/Metal)",
            os_name=os_name,
            cpu_name=cpu_info["name"],
            cpu_cores=cpu_info["cores"],
            torch_dtype="float16",
            optimizations=["torch_compile", "metal_shaders", "unified_memory"],
        )
    
    # --- 4. Check DirectML (Windows, any DX12 GPU) ---
    dml_info = _check_directml()
    if dml_info and dml_info["available"]:
        return HardwareInfo(
            backend="directml",
            device=dml_info["device"],
            device_name=dml_info["name"],
            os_name=os_name,
            cpu_name=cpu_info["name"],
            cpu_cores=cpu_info["cores"],
            torch_dtype="float32",  # DirectML often works better with fp32
            optimizations=["directml_ep"],
        )
    
    # --- 5. Check ROCm (AMD GPU on Linux) ---
    if cuda_info and cuda_info.get("is_rocm"):
        return HardwareInfo(
            backend="cuda",  # ROCm uses CUDA namespace
            device="cuda:0",
            device_name=f"{cuda_info['name']} (ROCm)",
            vram_gb=cuda_info["vram_gb"],
            os_name=os_name,
            cpu_name=cpu_info["name"],
            cpu_cores=cpu_info["cores"],
            torch_dtype="float16",
            optimizations=["rocm_hip"],
        )
    
    # --- 6. CPU Fallback ---
    optimizations = []
    if cpu_info["has_avx512"]:
        optimizations.append("avx512")
    elif cpu_info["has_avx2"]:
        optimizations.append("avx2")
    optimizations.append("int8_quantization")
    
    return HardwareInfo(
        backend="cpu",
        device="cpu",
        device_name=cpu_info["name"],
        os_name=os_name,
        cpu_name=cpu_info["name"],
        cpu_cores=cpu_info["cores"],
        has_avx2=cpu_info["has_avx2"],
        has_avx512=cpu_info["has_avx512"],
        torch_dtype="float32",
        optimizations=optimizations,
    )


def get_device_report(hw: HardwareInfo) -> str:
    """Generate a human-readable hardware report for server startup."""
    lines = [
        "",
        "=" * 55,
        "  [HW] HARDWARE DETECTION REPORT",
        "=" * 55,
        f"  OS:           {hw.os_name}",
        f"  CPU:          {hw.cpu_name} ({hw.cpu_cores} cores)",
    ]
    
    if hw.vram_gb:
        lines.append(f"  GPU:          {hw.device_name} ({hw.vram_gb} GB VRAM)")
    else:
        lines.append(f"  Accelerator:  {hw.device_name}")
    
    lines.extend([
        f"  Backend:      {hw.backend.upper()}",
        f"  Dtype:        {hw.torch_dtype}",
        f"  Optimizations: {', '.join(hw.optimizations) if hw.optimizations else 'none'}",
        "",
    ])
    
    # Performance estimate
    ttft_estimates = {
        "mlx": "~50-180ms (Apple Neural Engine)",
        "cuda": "~60-200ms (NVIDIA CUDA)",
        "mps": "~100-300ms (Apple Metal)",
        "directml": "~200-400ms (DirectX ML)",
        "cpu": "~500ms-2s (CPU only)",
    }
    lines.append(f"  Est. TTFT:    {ttft_estimates.get(hw.backend, 'unknown')}")
    lines.append("=" * 55)
    lines.append("")
    
    return "\n".join(lines)


# ─── Quick self-test ───
if __name__ == "__main__":
    hw = detect_hardware()
    print(get_device_report(hw))

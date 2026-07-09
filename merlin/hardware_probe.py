"""Hardware detection for Merlin AI.

Pure read-only probe. No side effects. No external dependencies beyond stdlib.
This is a Python port of the ram_gb() and hardware_tier() logic in scripts/doctor.sh
so that the FastAPI status layer can expose hardware context without shelling out.

Tier thresholds must stay in sync with:
  - configs/merlin/hardware-tiers.yaml
  - configs/merlin/model-tiers.env
  - scripts/doctor.sh::hardware_tier()

Tracked issue: #118 (model library / hardware-aware model guidance)
"""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tier thresholds — mirror hardware-tiers.yaml exactly
# ---------------------------------------------------------------------------

_TIERS: list[dict[str, Any]] = [
    {
        "tier": "high",
        "ram_gb_min": 48,
        "ram_gb_max": None,
        "max_model_class": "70b",
        "quantization": "q4/q5 or better",
        "default_profile": "core",
        "suggested_profiles": ["search", "automation", "coding", "ops"],
        "disabled_profiles": [],
        "recommended_models": ["llama3.3:70b", "qwen2.5:32b", "deepseek-r1:32b", "nomic-embed-text"],
        "warning": "High-memory tier: full stack available, but risky actions still require approval.",
    },
    {
        "tier": "mid",
        "ram_gb_min": 24,
        "ram_gb_max": 47,
        "max_model_class": "32b",
        "quantization": "q4/q5/q6",
        "default_profile": "core",
        "suggested_profiles": ["search", "automation"],
        "disabled_profiles": [],
        "recommended_models": ["qwen2.5:32b", "qwen2.5-coder:14b", "deepseek-r1:14b", "nomic-embed-text"],
        "warning": "Workstation tier: Magic Mode can run with approval gates.",
    },
    {
        "tier": "base",
        "ram_gb_min": 16,
        "ram_gb_max": 23,
        "max_model_class": "14b",
        "quantization": "q4/q5",
        "default_profile": "core",
        "suggested_profiles": ["search"],
        "disabled_profiles": [],
        "recommended_models": ["qwen2.5:7b", "qwen2.5-coder:7b", "deepseek-r1:7b", "nomic-embed-text"],
        "warning": "Developer laptop tier: run one heavy model or agent task at a time.",
    },
    {
        "tier": "low",
        "ram_gb_min": 8,
        "ram_gb_max": 15,
        "max_model_class": "7b",
        "quantization": "q4",
        "default_profile": "core",
        "suggested_profiles": [],
        "disabled_profiles": ["search", "automation", "coding", "security", "ops"],
        "recommended_models": ["qwen2.5:7b", "nomic-embed-text"],
        "warning": "Low-memory mode: avoid OpenHands, full search stack, and models above 7B.",
    },
    {
        "tier": "unsupported",
        "ram_gb_min": 0,
        "ram_gb_max": 7,
        "max_model_class": "none",
        "quantization": "none",
        "default_profile": "none",
        "suggested_profiles": [],
        "disabled_profiles": ["search", "automation", "coding", "security", "ops", "core"],
        "recommended_models": [],
        "warning": "RAM is below minimum supported tier (8 GB). Consider a hardware refresh.",
    },
]

_UNKNOWN_TIER: dict[str, Any] = {
    "tier": "unknown",
    "ram_gb_min": None,
    "ram_gb_max": None,
    "max_model_class": "7b",
    "quantization": "q4",
    "default_profile": "core",
    "suggested_profiles": [],
    "disabled_profiles": [],
    "recommended_models": ["qwen2.5:7b", "nomic-embed-text"],
    "warning": "RAM could not be detected. Defaulting to conservative 7B Q4 guidance.",
}


# ---------------------------------------------------------------------------
# RAM detection — stdlib only, mirrors doctor.sh logic
# ---------------------------------------------------------------------------

def _ram_bytes_macos() -> int | None:
    """Read hw.memsize via sysctl on macOS."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def _ram_bytes_linux() -> int | None:
    """Read MemTotal from /proc/meminfo on Linux."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, re.MULTILINE)
        if match:
            return int(match.group(1)) * 1024
    except Exception:
        pass
    return None


def ram_gb() -> int:
    """Return total physical RAM in whole GB. Returns 0 if detection fails."""
    system = platform.system()
    raw: int | None = None
    if system == "Darwin":
        raw = _ram_bytes_macos()
    elif system == "Linux":
        raw = _ram_bytes_linux()
    if raw and raw > 0:
        return raw // (1024 ** 3)
    return 0


# ---------------------------------------------------------------------------
# Tier resolution
# ---------------------------------------------------------------------------

def hardware_tier(ram: int) -> dict[str, Any]:
    """Return the tier definition dict for a given RAM value (GB).

    Returns the _UNKNOWN_TIER sentinel when ram == 0 (detection failure).
    """
    if ram <= 0:
        return _UNKNOWN_TIER
    for tier in _TIERS:
        min_gb = tier["ram_gb_min"]
        max_gb = tier["ram_gb_max"]
        if ram >= min_gb and (max_gb is None or ram <= max_gb):
            return tier
    return _UNKNOWN_TIER


# ---------------------------------------------------------------------------
# CPU / platform supplemental info (best-effort, no error on failure)
# ---------------------------------------------------------------------------

def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "physical_cores": None,
        "logical_cores": None,
        "architecture": platform.machine(),
        "avx512_support": None,
    }
    try:
        import os as _os
        info["logical_cores"] = _os.cpu_count()
    except Exception:
        pass

    system = platform.system()
    if system == "Linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            # physical cores via unique core id × physical id combos
            cores = set()
            phys_id = core_id = ""
            for line in cpuinfo.splitlines():
                if line.startswith("physical id"):
                    phys_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                    cores.add((phys_id, core_id))
            if cores:
                info["physical_cores"] = len(cores)
            info["avx512_support"] = "avx512f" in cpuinfo.lower()
        except Exception:
            pass
    elif system == "Darwin":
        try:
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                info["physical_cores"] = int(r.stdout.strip())
        except Exception:
            pass

    return info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hardware_profile() -> dict[str, Any]:
    """Return a complete, read-only hardware profile dict.

    This is the single source of truth for the /status/hardware endpoint.
    Zero side effects. Safe to call on every request.
    """
    detected_ram = ram_gb()
    tier_def = hardware_tier(detected_ram)

    return {
        "ram_gb": detected_ram,
        "tier": tier_def["tier"],
        "max_model_class": tier_def["max_model_class"],
        "quantization": tier_def["quantization"],
        "default_profile": tier_def["default_profile"],
        "suggested_profiles": tier_def["suggested_profiles"],
        "disabled_profiles": tier_def["disabled_profiles"],
        "recommended_models": tier_def["recommended_models"],
        "warning": tier_def["warning"],
        "cpu": _cpu_info(),
        "os": platform.system(),
        "detection_source": (
            "hw.memsize (sysctl)" if platform.system() == "Darwin"
            else "/proc/meminfo" if platform.system() == "Linux"
            else "unsupported_platform"
        ),
        "tier_config_source": "configs/merlin/hardware-tiers.yaml",
        "mode": "read_only_hardware_probe",
    }

"""CPU / hardware detection.

The engine adapts its research strategy to the machine it runs on
(weak-hardware mode). This module detects cores, SIMD width, cache sizes
and RAM without third-party dependencies.
"""

import os
import platform


def _cpuinfo_flags():
    flags = set()
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
    except OSError:
        pass
    return flags


def _cache_sizes():
    """Best-effort cache size detection (bytes) from /sys or /proc."""
    caches = {}
    try:
        base = "/sys/devices/system/cpu/cpu0/cache"
        for idx in os.listdir(base):
            try:
                with open(f"{base}/{idx}/level", "r") as f:
                    level = f.read().strip()
                with open(f"{base}/{idx}/size", "r") as f:
                    size = f.read().strip()  # e.g. "32K", "512K", "8M"
                mult = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}
                val = int(size[:-1]) * mult.get(size[-1].upper(), 1)
                key = f"L{level}"
                caches[key] = max(caches.get(key, 0), val)
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    return caches


def detect_hardware():
    flags = _cpuinfo_flags()
    simd = "none"
    for name in ("avx512f", "avx2", "avx", "sse4_2", "sse2"):
        if name in flags:
            simd = name
            break
    if not flags and platform.machine() == "x86_64":
        simd = "sse2 (assumed)"

    try:
        ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        ram_bytes = 0

    cores = os.cpu_count() or 1
    caches = _cache_sizes()

    # Weak-hardware classification drives research strategy.
    weak = cores <= 2 or (ram_bytes and ram_bytes < 5 * 1024 ** 3)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cores": cores,
        "simd": simd,
        "ram_bytes": ram_bytes,
        "ram_gb": round(ram_bytes / 1024 ** 3, 2) if ram_bytes else None,
        "caches": caches,
        "weak_hardware": weak,
        "numpy": _numpy_version(),
    }


def _numpy_version():
    try:
        import numpy
        return numpy.__version__
    except ImportError:
        return None


if __name__ == "__main__":
    import json
    print(json.dumps(detect_hardware(), indent=2))

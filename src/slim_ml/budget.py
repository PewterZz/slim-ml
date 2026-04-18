from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Tier(Enum):
    VRAM = "vram"
    RAM = "ram"
    NVME = "nvme"
    DISK = "disk"


@dataclass
class TierCapacity:
    total_bytes: int
    reserved_bytes: int = 0

    @property
    def available_bytes(self) -> int:
        return max(0, self.total_bytes - self.reserved_bytes)


class Budget(ABC):
    """Declares how much of each memory tier the runtime may use and tracks reservations.

    Interface designed so a future live-pressure implementation can be swapped in without
    touching call sites. Today's StaticBudget is a placeholder — see pressure signal risk
    in the design notes.
    """

    @abstractmethod
    def capacity(self, tier: Tier) -> TierCapacity: ...

    @abstractmethod
    def reserve(self, tier: Tier, nbytes: int, owner: str) -> bool: ...

    @abstractmethod
    def release(self, tier: Tier, nbytes: int, owner: str) -> None: ...

    def pressure(self, tier: Tier) -> float:
        cap = self.capacity(tier)
        if cap.total_bytes == 0:
            return 0.0
        return cap.reserved_bytes / cap.total_bytes


class StaticBudget(Budget):
    def __init__(self, limits: dict[Tier, int], headroom_bytes: dict[Tier, int] | None = None):
        headroom_bytes = headroom_bytes or {}
        self._caps = {
            t: TierCapacity(total_bytes=max(0, limits.get(t, 0) - headroom_bytes.get(t, 0)))
            for t in Tier
        }
        self._reservations: dict[tuple[Tier, str], int] = {}

    def capacity(self, tier: Tier) -> TierCapacity:
        return self._caps[tier]

    def reserve(self, tier: Tier, nbytes: int, owner: str) -> bool:
        cap = self._caps[tier]
        if cap.available_bytes < nbytes:
            return False
        cap.reserved_bytes += nbytes
        key = (tier, owner)
        self._reservations[key] = self._reservations.get(key, 0) + nbytes
        return True

    def release(self, tier: Tier, nbytes: int, owner: str) -> None:
        key = (tier, owner)
        current = self._reservations.get(key, 0)
        freed = min(current, nbytes)
        self._reservations[key] = current - freed
        self._caps[tier].reserved_bytes = max(0, self._caps[tier].reserved_bytes - freed)


def auto_detect_limits(headroom_bytes: Optional[int] = None) -> dict[Tier, int]:
    """Best-effort hardware detection. Returns per-tier byte limits.

    headroom_bytes: bytes to leave unclaimed in RAM for the user's other work.
    Conservative default on Macs (unified memory) is important.
    """
    import platform
    import shutil

    import psutil

    total_ram = psutil.virtual_memory().total
    if headroom_bytes is None:
        headroom_bytes = 4 * 1024**3 if platform.system() == "Darwin" else 2 * 1024**3

    limits: dict[Tier, int] = {
        Tier.RAM: max(0, total_ram - headroom_bytes),
        Tier.VRAM: 0,
        Tier.NVME: 0,
        Tier.DISK: shutil.disk_usage("/").free,
    }
    try:
        import subprocess

        if platform.system() == "Linux":
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False, timeout=2,
            )
            if out.returncode == 0 and out.stdout.strip():
                mib = int(out.stdout.strip().splitlines()[0])
                limits[Tier.VRAM] = mib * 1024 * 1024
    except Exception:
        pass
    return limits

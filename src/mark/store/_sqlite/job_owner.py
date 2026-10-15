"""Who holds a fact-extraction job, and whether that holder is still alive.

A claimed job records an *owner* token for the claiming process:

    v1|<host>|<boot id>|<pid>|<process start time>

so another process opening the same database can tell, without waiting for
the job lease to expire, that the holder crashed (kill -9, OOM, power cut)
and reclaim its jobs immediately. Liveness is only ever judged for owners on
*this* host: a pid means nothing on another machine sharing the file, so
those jobs fall back to the lease. Anything we cannot decide (unknown
platform, permission errors) counts as alive, because wrongly stealing a
running job costs duplicate work while waiting for the lease only costs time.

The boot id catches reboots and the process start time catches pid reuse
(both are read from /proc on Linux and simply left empty elsewhere).
Standard library only.
"""
from __future__ import annotations

import os
import socket
import sys
from functools import lru_cache

_PREFIX = "v1"


def _read(path: str) -> str:
    try:
        with open(path, encoding="ascii", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


@lru_cache(maxsize=1)
def _host() -> str:
    # "|" separates fields; a hostname never contains it, but be safe.
    return socket.gethostname().replace("|", "_")


@lru_cache(maxsize=1)
def _boot_id() -> str:
    return _read("/proc/sys/kernel/random/boot_id")


def _start_time(pid: int) -> str:
    """Kernel start time of *pid* (clock ticks since boot), or "" when unknown."""
    stat = _read(f"/proc/{pid}/stat")
    if not stat:
        return ""
    # The command name (field 2) may contain spaces/parentheses: split after its last ")".
    fields = stat.rpartition(")")[2].split()
    return fields[19] if len(fields) > 19 else ""


def current_owner() -> str:
    """Owner token for the calling process (recomputed after a fork, since the pid changes)."""
    pid = os.getpid()
    return "|".join((_PREFIX, _host(), _boot_id(), str(pid), _start_time(pid)))


def owner_is_dead(owner: str | None) -> bool:
    """True only when *owner* certainly no longer runs (same host, process gone)."""
    if not owner:
        return False
    parts = owner.split("|")
    if len(parts) != 5 or parts[0] != _PREFIX or parts[1] != _host():
        return False
    _, _, boot, pid_text, started = parts
    if boot and _boot_id() and boot != _boot_id():
        return True  # the machine rebooted since the job was claimed
    try:
        pid = int(pid_text)
    except ValueError:
        return False
    if pid == os.getpid():
        return False
    if not _pid_alive(pid):
        return True
    # Same pid, different start time: the pid was reused by another process.
    now_started = _start_time(pid)
    return bool(started and now_started and now_started != started)


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        # os.kill(pid, 0) would *terminate* the process on Windows; rely on the lease there.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, ValueError):
        return True
    return True

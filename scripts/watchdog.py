#!/usr/bin/env python3
"""Apple-Silicon thermal + memory-pressure watchdog for long training runs.

Polls macOS's native thermal-pressure and memory-pressure signals every N seconds.
If either crosses a danger threshold, sends SIGTERM to the target training
process (whose PID is found by pgrep on a substring of its command line),
then SIGKILL after 30 s if the process is still alive. Logs every sample.

Requires only stdlib + macOS-native CLIs (`pmset`, `sysctl`, `pgrep`). No sudo.

Usage (in a second terminal, while training is running in the first):

    .venv/bin/python scripts/watchdog.py --target src/q8_roberta.py

Customize thresholds:

    .venv/bin/python scripts/watchdog.py \\
        --target src/q8_roberta.py \\
        --interval 20 \\
        --max-thermal Serious \\
        --max-mem-pressure 2 \\
        --log watchdog-q8.log

Why these thresholds by default:
- Thermal: `CPU_Speed_Limit` between 70–89 → "Fair" (mild throttling, fine to
  continue). 50–69 → "Serious" (sustained throttle, training will slow markedly
  and the chassis is hot). <50 → "Critical" (kernel is hard-throttling; stop).
  Default trip = `Serious`.
- Memory: kernel pressure level 1 = normal, 2 = warning (some swap activity but
  manageable), 4 = critical (system is paging hard and approaching OOM).
  Default trip = `2` (warning). For 24/64/128 GB unified-memory Macs running a
  3-seed BERT/RoBERTa fine-tune at batch_size=8 you should never see >1.

Exit codes:
- 0  target process exited on its own (training completed); watchdog stood down.
- 1  watchdog killed the target after a breach.
- 2  argument error.
- 3  hard error reading sensors (cannot proceed).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Thermal pressure levels by descending CPU_Speed_Limit thresholds.
THERMAL_LEVELS = ("Nominal", "Fair", "Serious", "Critical")
THERMAL_RANK = {name: i for i, name in enumerate(THERMAL_LEVELS)}


def get_thermal_state() -> tuple[str, int | None]:
    """Return (level, cpu_speed_limit) where level ∈ THERMAL_LEVELS."""
    try:
        out = subprocess.check_output(
            ["pmset", "-g", "therm"], text=True, stderr=subprocess.DEVNULL
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ("Unknown", None)
    limit: int | None = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("CPU_Speed_Limit"):
            try:
                limit = int(s.split("=", 1)[1].strip())
            except ValueError:
                pass
    if limit is None:
        return ("Nominal", None)
    if limit >= 90:
        return ("Nominal", limit)
    if limit >= 70:
        return ("Fair", limit)
    if limit >= 50:
        return ("Serious", limit)
    return ("Critical", limit)


def get_memory_pressure() -> int:
    """Return the kernel's memorystatus_vm_pressure_level (1, 2, or 4)."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(out)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return -1


def find_pid(target_substring: str) -> int | None:
    """Return the first PID whose command line matches `target_substring`."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", target_substring], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return None
    if not out:
        return None
    pids = [int(line) for line in out.splitlines() if line.strip().isdigit()]
    # exclude our own pid in case the substring matched our argv
    pids = [p for p in pids if p != os.getpid()]
    return pids[0] if pids else None


def graceful_kill(pid: int, hard_after_seconds: int = 30) -> None:
    """SIGTERM first; SIGKILL after `hard_after_seconds` if still alive."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    print(f"sent SIGTERM to pid={pid}; waiting up to {hard_after_seconds}s for graceful exit")
    deadline = time.monotonic() + hard_after_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"pid={pid} exited gracefully")
            return
        time.sleep(1)
    print(f"pid={pid} survived SIGTERM; sending SIGKILL")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target",
        required=True,
        help='Substring matched against `pgrep -f` (e.g. "src/q8_roberta.py").',
    )
    parser.add_argument(
        "--interval", type=int, default=30, help="Poll interval in seconds (default 30)."
    )
    parser.add_argument(
        "--max-thermal",
        default="Serious",
        choices=("Fair", "Serious", "Critical"),
        help="Thermal level (inclusive) at which to kill the target.",
    )
    parser.add_argument(
        "--max-mem-pressure",
        type=int,
        default=2,
        choices=(2, 4),
        help="Kernel memory-pressure level at which to kill (2=warning, 4=critical).",
    )
    parser.add_argument("--log", default="watchdog.log", help="Append-only log path.")
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="Exit with error if target is not already running at startup.",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    trip_thermal_rank = THERMAL_RANK[args.max_thermal]

    print(f"Watchdog started — polling every {args.interval}s, logging to {log_path}.")
    print(
        f"Trip rules: thermal ≥ {args.max_thermal!r}  OR  "
        f"memory_pressure ≥ {args.max_mem_pressure}"
    )
    print(f"Target substring: {args.target!r}")
    print("─" * 78)

    initial_pid = find_pid(args.target)
    if initial_pid is None:
        msg = f"No process matching {args.target!r} found at startup."
        if args.require_target:
            print(f"{msg} Exiting (–-require-target was set).")
            return 2
        print(f"{msg} Will wait for it to start.")

    saw_target_alive = initial_pid is not None

    try:
        while True:
            now = datetime.now().strftime("%H:%M:%S")
            therm_level, cpu_limit = get_thermal_state()
            mem_level = get_memory_pressure()
            pid = find_pid(args.target)

            cpu_str = f"{cpu_limit:3d}%" if cpu_limit is not None else "  ? "
            pid_str = f"pid={pid}" if pid is not None else "pid=none"
            line = (
                f"[{now}] thermal={therm_level:<8s} (CPU_Speed_Limit={cpu_str})  "
                f"mem_pressure={mem_level}  {pid_str}"
            )
            print(line, flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")

            if pid is None:
                if saw_target_alive:
                    print("Target process is no longer running. Watchdog standing down.")
                    return 0
            else:
                saw_target_alive = True

            breaches: list[str] = []
            if therm_level != "Unknown" and THERMAL_RANK[therm_level] >= trip_thermal_rank:
                breaches.append(f"thermal={therm_level} (CPU_Speed_Limit={cpu_limit})")
            if mem_level >= args.max_mem_pressure:
                breaches.append(f"memory_pressure={mem_level}")

            if breaches and pid is not None:
                bullet = "; ".join(breaches)
                print("─" * 78)
                print(f"⚠  THRESHOLD BREACHED: {bullet}")
                with log_path.open("a") as f:
                    f.write(f"BREACH  {bullet}\n")
                graceful_kill(pid)
                return 1

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nWatchdog interrupted by user; target left running.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

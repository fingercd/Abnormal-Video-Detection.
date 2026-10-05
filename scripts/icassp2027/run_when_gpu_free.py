#!/usr/bin/env python3
"""Start one command after a verified GPU satisfies its explicit resource policy.

This is deliberately a launcher rather than a scheduler.  It never submits a job,
changes an environment, or retries a completed child process.  A GPU is eligible
by default only when ``nvidia-smi`` reports low used memory and no compute PID.
An explicitly authorized sharing mode instead requires minimum free memory;
both policies recheck after taking one advisory lease for the physical UUID.
Sharing is for ordinary workloads and is never eligible for formal timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

try:  # The launcher itself is Linux-only; keeping imports portable aids unit tests.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    fcntl = None  # type: ignore[assignment]


MAX_IDLE_MEMORY_MIB = 1024
RUNNING_POLL_SECONDS = 2
SHARING_OBSERVATION_SECONDS = 60
TERMINATE_GRACE_SECONDS = 10
KILL_GRACE_SECONDS = 10
KILL_SIGNAL = getattr(signal, "SIGKILL", 9)
SENSITIVE_WORDS = ("token", "secret", "password", "cookie", "apikey", "api-key", "private")


class GuardError(RuntimeError):
    """A condition that must prevent a child process from being started."""


@dataclass(frozen=True)
class GpuSnapshot:
    index: int
    uuid: str
    memory_used_mib: int
    compute_pids: frozenset[int]
    memory_free_mib: int | None = None

    @property
    def idle(self) -> bool:
        return not self.compute_pids and self.memory_used_mib <= MAX_IDLE_MEMORY_MIB


class ProcessLike(Protocol):
    pid: int

    def poll(self) -> int | None: ...


class Lease(Protocol):
    def release(self) -> None: ...


class FcntlLease:
    def __init__(self, path: Path) -> None:
        if fcntl is None:
            raise GuardError("GPU leases require Linux fcntl support")
        self._file = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file.close()
            raise

    def release(self) -> None:
        if fcntl is None:  # pragma: no cover - constructor already refuses this path
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()


def _run_checked(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command), check=False, capture_output=True, text=True, encoding="utf-8"
        )
    except OSError as exc:
        raise GuardError(f"could not execute {command[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise GuardError(f"command failed ({command[0]}): {detail}")
    return completed.stdout


def _csv_rows(text: str, source: str, expected_fields: int) -> list[list[str]]:
    try:
        rows = list(csv.reader(line for line in text.splitlines() if line.strip()))
    except csv.Error as exc:
        raise GuardError(f"cannot parse {source} CSV") from exc
    if any(len(row) != expected_fields for row in rows):
        raise GuardError(f"unexpected {source} CSV fields")
    return [[field.strip() for field in row] for row in rows]


def probe_gpu(
    index: int, command_runner: Callable[[Sequence[str]], str] = _run_checked
) -> GpuSnapshot:
    """Read one physical GPU and its compute PIDs; unknown output is unsafe."""

    gpu_rows = _csv_rows(
        command_runner(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ]
        ),
        "GPU",
        4,
    )
    matching = [row for row in gpu_rows if row[0] == str(index)]
    if len(matching) != 1:
        raise GuardError(f"GPU index {index} was not uniquely reported")
    _, uuid, memory_text, free_text = matching[0]
    try:
        memory_used_mib = int(memory_text)
        memory_free_mib = int(free_text)
    except ValueError as exc:
        raise GuardError(f"GPU {index} reported an invalid memory value") from exc
    if not uuid or memory_used_mib < 0 or memory_free_mib < 0:
        raise GuardError(f"GPU {index} reported invalid identity data")

    pid_rows = _csv_rows(
        command_runner(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        ),
        "compute-process",
        2,
    )
    pids: set[int] = set()
    for gpu_uuid, pid_text in pid_rows:
        if gpu_uuid != uuid:
            continue
        try:
            pid = int(pid_text)
        except ValueError as exc:
            raise GuardError(f"GPU {index} reported an invalid compute PID") from exc
        if pid <= 0:
            raise GuardError(f"GPU {index} reported a non-positive compute PID")
        pids.add(pid)
    return GpuSnapshot(
        index=index, uuid=uuid, memory_used_mib=memory_used_mib, compute_pids=frozenset(pids),
        memory_free_mib=memory_free_mib,
    )


def verify_worktree(
    cwd: Path,
    expected_commit: str,
    command_runner: Callable[[Sequence[str]], str] = _run_checked,
) -> None:
    if not cwd.is_dir():
        raise GuardError(f"cwd is not a directory: {cwd}")
    head = command_runner(["git", "-C", str(cwd), "rev-parse", "HEAD"]).strip()
    if head != expected_commit:
        raise GuardError(f"Git HEAD is {head!r}, expected {expected_commit!r}")
    # Untracked and ignored local assets are intentionally omitted.  Any staged,
    # modified, renamed, or deleted tracked path blocks launch.
    tracked_changes = command_runner(
        ["git", "-C", str(cwd), "status", "--porcelain", "--untracked-files=no"]
    ).strip()
    if tracked_changes:
        raise GuardError("tracked worktree changes are present")


def redact_command(argv: Sequence[str]) -> list[str]:
    """Keep command provenance without persisting obvious credential values."""

    redacted: list[str] = []
    hide_next = False
    for item in argv:
        lower = item.lower()
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
        elif any(word in lower for word in SENSITIVE_WORDS):
            if "=" in item:
                key, _ = item.split("=", 1)
                redacted.append(f"{key}=<redacted>")
            else:
                redacted.append(item)
                hide_next = item.startswith("-")
        else:
            redacted.append(item)
    return redacted


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _status(output_dir: Path, state: str, **details: Any) -> None:
    _write_json(
        output_dir / "status.json",
        {"state": state, "updated_at_epoch": time.time(), **details},
    )


def _lease_path(lease_dir: Path, uuid: str) -> Path:
    safe_uuid = uuid.replace("/", "_")
    return lease_dir / f"gpu-{safe_uuid}.lease"


def acquire_lease(lease_dir: Path, uuid: str) -> Lease | None:
    lease_dir.mkdir(parents=True, exist_ok=True)
    try:
        return FcntlLease(_lease_path(lease_dir, uuid))
    except BlockingIOError:
        return None


def _candidate(
    indices: Iterable[int],
    probe: Callable[[int], GpuSnapshot],
    lease_factory: Callable[[str], Lease | None],
    *,
    allow_sharing: bool = False,
    min_free_memory_mib: int = 16384,
) -> tuple[GpuSnapshot, Lease] | None:
    def eligible(snapshot: GpuSnapshot) -> bool:
        if allow_sharing:
            return (
                snapshot.memory_free_mib is not None
                and snapshot.memory_free_mib >= min_free_memory_mib
            )
        return snapshot.idle

    for index in indices:
        first = probe(index)
        if not eligible(first):
            continue
        lease = lease_factory(first.uuid)
        if lease is None:
            continue
        try:
            second = probe(index)
            # A changed UUID means the index cannot safely be trusted.  Requiring
            # the same UUID also binds CUDA_VISIBLE_DEVICES to the leased card.
            if second.uuid == first.uuid and eligible(second):
                return second, lease
        except Exception:
            lease.release()
            raise
        lease.release()
    return None


def _sharing_observation(snapshot: GpuSnapshot, *, phase: str) -> dict[str, Any]:
    processes = []
    for pid in sorted(snapshot.compute_pids):
        owner = uid = None
        try:
            import pwd

            uid = (Path("/proc") / str(pid)).stat().st_uid
            owner = pwd.getpwuid(uid).pw_name
        except (ImportError, FileNotFoundError, ProcessLookupError, PermissionError, KeyError):
            # The original PID remains in the observation even if it exits
            # before its owner can be resolved. It never becomes an idle PID set.
            pass
        processes.append({"pid": pid, "owner": owner, "uid": uid})
    return {
        "phase": phase,
        "observed_at_epoch": time.time(),
        "index": snapshot.index,
        "uuid": snapshot.uuid,
        "memory_used_mib": snapshot.memory_used_mib,
        "memory_free_mib": snapshot.memory_free_mib,
        "compute_pids": sorted(snapshot.compute_pids),
        "compute_processes": processes,
    }


def _append_sharing_observation(output_dir: Path, observation: dict[str, Any]) -> None:
    with (output_dir / "sharing-observations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(observation, sort_keys=True) + "\n")


def _wait_for_exit(
    process: ProcessLike,
    *,
    seconds: int,
    poll_seconds: int,
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
) -> int | None:
    deadline = monotonic() + seconds
    while True:
        exit_code = process.poll()
        if exit_code is not None:
            return exit_code
        remaining = deadline - monotonic()
        if remaining <= 0:
            return None
        sleep(min(poll_seconds, remaining))


def _terminate_until_exited(
    process: ProcessLike,
    *,
    requested_state: str,
    output_dir: Path,
    metadata: dict[str, Any],
    killpg: Callable[[int, int], None],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    running_poll_seconds: int,
    terminate_grace_seconds: int,
    kill_grace_seconds: int,
) -> tuple[int, str]:
    """Stop only our process group and do not return until its child exits."""

    _status(output_dir, "terminating", requested_state=requested_state, pid=process.pid, **metadata)
    signal_name = "SIGTERM"
    with suppress(ProcessLookupError):
        killpg(process.pid, signal.SIGTERM)
    exit_code = _wait_for_exit(
        process,
        seconds=terminate_grace_seconds,
        poll_seconds=running_poll_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    if exit_code is not None:
        return exit_code, signal_name

    signal_name = "SIGKILL"
    with suppress(ProcessLookupError):
        killpg(process.pid, KILL_SIGNAL)
    exit_code = _wait_for_exit(
        process,
        seconds=kill_grace_seconds,
        poll_seconds=running_poll_seconds,
        sleep=sleep,
        monotonic=monotonic,
    )
    if exit_code is not None:
        return exit_code, signal_name

    # The process survived both signals.  Retain the lease and keep watching so
    # another launcher cannot claim this GPU while our child may still own it.
    _status(
        output_dir,
        "termination_pending",
        requested_state=requested_state,
        pid=process.pid,
        termination_signal=signal_name,
        **metadata,
    )
    while True:
        exit_code = process.poll()
        if exit_code is not None:
            return exit_code, signal_name
        sleep(running_poll_seconds)


def _posix_getpgid(pid: int) -> int:
    if not hasattr(os, "getpgid"):
        raise GuardError("GPU contention checks require POSIX process groups")
    return os.getpgid(pid)


def _posix_killpg(process_group: int, signum: int) -> None:
    if not hasattr(os, "killpg"):
        raise GuardError("GPU contention checks require POSIX process groups")
    os.killpg(process_group, signum)


def _external_compute_pids(
    snapshot: GpuSnapshot,
    process_group: int,
    getpgid: Callable[[int], int],
) -> set[int]:
    external: set[int] = set()
    for pid in snapshot.compute_pids:
        try:
            if getpgid(pid) != process_group:
                external.add(pid)
        except ProcessLookupError as exc:
            # The process list changed while inspecting it.  It is no longer
            # possible to prove that no outside process owns this card.
            raise GuardError("compute PID disappeared during contention check") from exc
    return external


def run_guard(
    *,
    cwd: Path,
    expected_commit: str,
    output_dir: Path,
    lease_dir: Path,
    gpu_indices: Sequence[int],
    poll_seconds: int,
    max_wait_seconds: int,
    child_argv: Sequence[str],
    allow_sharing: bool = False,
    min_free_memory_mib: int = 16384,
    formal_timing: bool = False,
    probe: Callable[[int], GpuSnapshot] = probe_gpu,
    lease_factory: Callable[[str], Lease | None] | None = None,
    popen: Callable[..., ProcessLike] = subprocess.Popen,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    getpgid: Callable[[int], int] | None = None,
    killpg: Callable[[int, int], None] | None = None,
    hostname: Callable[[], str] = socket.gethostname,
    running_poll_seconds: int = RUNNING_POLL_SECONDS,
    terminate_grace_seconds: int = TERMINATE_GRACE_SECONDS,
    kill_grace_seconds: int = KILL_GRACE_SECONDS,
) -> str:
    """Run the one permitted child process, returning its terminal state."""

    if poll_seconds < 30:
        raise GuardError("poll_seconds must be at least 30")
    if max_wait_seconds < 0:
        raise GuardError("max_wait_seconds must not be negative")
    if running_poll_seconds <= 0 or terminate_grace_seconds <= 0 or kill_grace_seconds <= 0:
        raise GuardError("running and termination poll intervals must be positive")
    if not gpu_indices:
        raise GuardError("at least one gpu-index is required")
    if len(set(gpu_indices)) != len(gpu_indices):
        raise GuardError("gpu-index values must be unique")
    if not child_argv:
        raise GuardError("a child command after -- is required")
    if min_free_memory_mib <= 0:
        raise GuardError("min_free_memory_mib must be positive")
    if allow_sharing and formal_timing:
        raise GuardError("formal timing requires exclusive GPU mode; sharing is not allowed")
    if output_dir.exists():
        raise GuardError(f"output-dir must be new: {output_dir}")

    # Resolve POSIX-only functions only when the launcher is actually invoked.
    # This lets unit tests import the module on Windows and inject fakes, while a
    # real non-POSIX launch still fails before it can start a child process.
    if (getpgid is None or killpg is None) and (
        not hasattr(os, "getpgid") or not hasattr(os, "killpg")
    ):
        raise GuardError("GPU contention checks require a POSIX host")
    getpgid = getpgid or _posix_getpgid
    killpg = killpg or _posix_killpg
    verify_worktree(cwd.resolve(), expected_commit)
    output_dir.mkdir(parents=True)
    lease_factory = lease_factory or (lambda uuid: acquire_lease(lease_dir, uuid))
    started_wait = monotonic()
    cancel_file = output_dir / "cancel"
    policy = {
        "allow_sharing": allow_sharing,
        "min_free_memory_mib": min_free_memory_mib if allow_sharing else None,
        "formal_timing_requested": formal_timing,
        "formal_timing_eligible": not allow_sharing,
    }
    _status(
        output_dir, "waiting", gpu_indices=list(gpu_indices),
        max_wait_seconds=max_wait_seconds, **policy,
    )

    while True:
        if cancel_file.exists():
            _status(output_dir, "cancelled", phase="waiting", **policy)
            return "cancelled"
        if monotonic() - started_wait >= max_wait_seconds:
            _status(output_dir, "timed_out", phase="waiting", **policy)
            return "timed_out"
        try:
            selected = _candidate(
                gpu_indices, probe, lease_factory,
                allow_sharing=allow_sharing, min_free_memory_mib=min_free_memory_mib,
            )
        except GuardError as exc:
            _status(output_dir, "failed_preflight", reason=str(exc), **policy)
            return "failed_preflight"
        if selected is None:
            _status(
                output_dir, "waiting",
                reason="no leased GPU passed both resource-policy checks", **policy,
            )
            sleep(poll_seconds)
            continue

        snapshot, lease = selected
        try:
            try:
                verify_worktree(cwd.resolve(), expected_commit)
            except GuardError as exc:
                _status(
                    output_dir, "failed_preflight", reason=str(exc),
                    gpu_uuid=snapshot.uuid, **policy,
                )
                return "failed_preflight"
            command = [item.replace("{device}", "cuda:0") for item in child_argv]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = snapshot.uuid
            log_path = output_dir / "launcher.log"
            metadata = {
                "host": hostname(),
                "gpu_index": snapshot.index,
                "gpu_uuid": snapshot.uuid,
                "gpu_memory_used_mib_before_start": snapshot.memory_used_mib,
                "gpu_memory_free_mib_before_start": snapshot.memory_free_mib,
                "gpu_compute_pids_before_start": sorted(snapshot.compute_pids),
                "idle_memory_limit_mib": None if allow_sharing else MAX_IDLE_MEMORY_MIB,
                **policy,
                "command": redact_command(command),
                "environment": {
                    key: environment[key]
                    for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
                    if key in environment
                },
            }
            last_observation_at = monotonic()
            last_observed_pids = snapshot.compute_pids
            if allow_sharing:
                observation = _sharing_observation(snapshot, phase="before_start")
                metadata["initial_gpu_observation"] = observation
                metadata["latest_gpu_observation"] = observation
                _append_sharing_observation(output_dir, observation)
            try:
                with log_path.open("x", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
                    log_file.flush()
                    process = popen(
                        command,
                        cwd=str(cwd),
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        shell=False,
                    )
            except OSError as exc:
                _status(
                    output_dir,
                    "failed_launch",
                    reason=f"child process could not be created: {exc}",
                    **metadata,
                )
                return "failed_launch"
            (output_dir / "pid").write_text(f"{process.pid}\n", encoding="utf-8")
            _status(output_dir, "running", pid=process.pid, **metadata)
            while True:
                exit_code = process.poll()
                if exit_code is not None:
                    state = "completed" if exit_code == 0 else "failed_exit"
                    _status(output_dir, state, pid=process.pid, exit_code=exit_code, **metadata)
                    return state
                if cancel_file.exists():
                    exit_code, signal_name = _terminate_until_exited(
                        process,
                        requested_state="cancelled",
                        output_dir=output_dir,
                        metadata=metadata,
                        killpg=killpg,
                        sleep=sleep,
                        monotonic=monotonic,
                        running_poll_seconds=running_poll_seconds,
                        terminate_grace_seconds=terminate_grace_seconds,
                        kill_grace_seconds=kill_grace_seconds,
                    )
                    _status(
                        output_dir,
                        "cancelled",
                        phase="running",
                        pid=process.pid,
                        exit_code=exit_code,
                        termination_signal=signal_name,
                        **metadata,
                    )
                    return "cancelled"
                try:
                    current = probe(snapshot.index)
                    if current.uuid != snapshot.uuid:
                        raise GuardError("GPU index no longer resolves to the leased UUID")
                    external = (
                        set() if allow_sharing
                        else _external_compute_pids(current, process.pid, getpgid)
                    )
                except GuardError as exc:
                    exit_code, signal_name = _terminate_until_exited(
                        process,
                        requested_state="interrupted_contention",
                        output_dir=output_dir,
                        metadata=metadata,
                        killpg=killpg,
                        sleep=sleep,
                        monotonic=monotonic,
                        running_poll_seconds=running_poll_seconds,
                        terminate_grace_seconds=terminate_grace_seconds,
                        kill_grace_seconds=kill_grace_seconds,
                    )
                    _status(
                        output_dir,
                        "interrupted_contention",
                        pid=process.pid,
                        reason=str(exc),
                        exit_code=exit_code,
                        termination_signal=signal_name,
                        **metadata,
                    )
                    return "interrupted_contention"
                if allow_sharing and (
                    current.compute_pids != last_observed_pids
                    or monotonic() - last_observation_at >= SHARING_OBSERVATION_SECONDS
                ):
                    observation = _sharing_observation(current, phase="running")
                    metadata["latest_gpu_observation"] = observation
                    _append_sharing_observation(output_dir, observation)
                    _status(output_dir, "running", pid=process.pid, **metadata)
                    last_observation_at = monotonic()
                    last_observed_pids = current.compute_pids
                if external:
                    exit_code, signal_name = _terminate_until_exited(
                        process,
                        requested_state="interrupted_contention",
                        output_dir=output_dir,
                        metadata=metadata,
                        killpg=killpg,
                        sleep=sleep,
                        monotonic=monotonic,
                        running_poll_seconds=running_poll_seconds,
                        terminate_grace_seconds=terminate_grace_seconds,
                        kill_grace_seconds=kill_grace_seconds,
                    )
                    _status(
                        output_dir,
                        "interrupted_contention",
                        pid=process.pid,
                        external_compute_pids=sorted(external),
                        exit_code=exit_code,
                        termination_signal=signal_name,
                        **metadata,
                    )
                    return "interrupted_contention"
                sleep(running_poll_seconds)
        finally:
            lease.release()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lease-dir", required=True, type=Path)
    parser.add_argument("--gpu-index", required=True, type=int, action="append")
    parser.add_argument("--poll-seconds", required=True, type=int)
    parser.add_argument("--max-wait-seconds", required=True, type=int)
    parser.add_argument(
        "--allow-sharing", action="store_true",
        help="explicitly allow existing compute processes; never use for formal timing",
    )
    parser.add_argument("--min-free-memory-mib", type=int, default=16384)
    parser.add_argument("--formal-timing", action="store_true", help="require exclusive GPU mode")
    parser.add_argument("child_argv", nargs=argparse.REMAINDER, help="command after --")
    args = parser.parse_args(argv)
    if args.child_argv[:1] == ["--"]:
        args.child_argv = args.child_argv[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        state = run_guard(
            cwd=args.cwd,
            expected_commit=args.expected_commit,
            output_dir=args.output_dir,
            lease_dir=args.lease_dir,
            gpu_indices=args.gpu_index,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
            child_argv=args.child_argv,
            allow_sharing=args.allow_sharing,
            min_free_memory_mib=args.min_free_memory_mib,
            formal_timing=args.formal_timing,
        )
    except GuardError as exc:
        print(f"gpu guard refused to start: {exc}", file=sys.stderr)
        return 2
    return 0 if state == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

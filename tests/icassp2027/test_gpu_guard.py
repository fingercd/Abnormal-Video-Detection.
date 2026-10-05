from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "icassp2027" / "run_when_gpu_free.py"
SPEC = importlib.util.spec_from_file_location("run_when_gpu_free", SCRIPT)
assert SPEC and SPEC.loader
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)


class FakeLease:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class FakeProcess:
    def __init__(self, pid: int, answers: list[int | None]) -> None:
        self.pid = pid
        self._answers = iter(answers)

    def poll(self) -> int | None:
        return next(self._answers)


def snapshots(*items: guard.GpuSnapshot):
    values = iter(items)
    return lambda _index: next(values)


def ready_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, str]:
    cwd = tmp_path / "checkout"
    cwd.mkdir()
    expected = "a" * 40
    monkeypatch.setattr(guard, "verify_worktree", lambda actual, commit: (actual, commit))
    return cwd, expected


def test_probe_gpu_requires_parseable_empty_compute_listing() -> None:
    replies = iter(
        [
            "0, GPU-test, 42, 40000\n1, GPU-other, 900, 32000\n",
            "",
        ]
    )

    snapshot = guard.probe_gpu(0, command_runner=lambda _command: next(replies))

    assert snapshot == guard.GpuSnapshot(0, "GPU-test", 42, frozenset(), 40000)

    bad_replies = iter(["0, GPU-test, 42, 40000\n", "GPU-test, not-a-pid\n"])
    with pytest.raises(guard.GuardError, match="invalid compute PID"):
        guard.probe_gpu(0, command_runner=lambda _command: next(bad_replies))


def test_parser_replaces_device_and_checks_gpu_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cwd, expected = ready_guard(monkeypatch, tmp_path)
    worktree_checks: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        guard,
        "verify_worktree",
        lambda actual, commit: worktree_checks.append((actual, commit)),
    )
    first = guard.GpuSnapshot(3, "GPU-test", 12, frozenset())
    second = guard.GpuSnapshot(3, "GPU-test", 12, frozenset())
    seen: dict[str, object] = {}
    lease = FakeLease()

    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return FakeProcess(4321, [0])

    state = guard.run_guard(
        cwd=cwd,
        expected_commit=expected,
        output_dir=tmp_path / "out",
        lease_dir=tmp_path / "leases",
        gpu_indices=[3],
        poll_seconds=30,
        max_wait_seconds=60,
        child_argv=["python", "train.py", "--device", "{device}"],
        probe=snapshots(first, second),
        lease_factory=lambda _uuid: lease,
        popen=fake_popen,
        getpgid=lambda pid: pid,
        killpg=lambda _pgid, _signum: None,
    )

    assert state == "completed"
    assert seen["argv"] == ["python", "train.py", "--device", "cuda:0"]
    assert seen["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    assert seen["kwargs"]["shell"] is False
    assert seen["kwargs"]["start_new_session"] is True
    assert lease.released
    assert worktree_checks == [(cwd.resolve(), expected), (cwd.resolve(), expected)]
    status = json.loads((tmp_path / "out" / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "completed"


def test_compute_pid_blocks_launch_even_when_memory_is_low(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cwd, expected = ready_guard(monkeypatch, tmp_path)
    unsafe = guard.GpuSnapshot(0, "GPU-busy", 12, frozenset({99}), 40000)
    launched = False
    ticks = iter([0.0, 0.0, 1.0])

    def fake_popen(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("must not launch")

    state = guard.run_guard(
        cwd=cwd,
        expected_commit=expected,
        output_dir=tmp_path / "out",
        lease_dir=tmp_path / "leases",
        gpu_indices=[0],
        poll_seconds=30,
        max_wait_seconds=1,
        child_argv=["python", "train.py"],
        probe=lambda _index: unsafe,
        popen=fake_popen,
        monotonic=lambda: next(ticks),
        sleep=lambda _seconds: None,
        getpgid=lambda pid: pid,
        killpg=lambda _pgid, _signum: None,
    )

    assert state == "timed_out"
    assert not launched


def test_invalid_probe_data_never_launches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cwd, expected = ready_guard(monkeypatch, tmp_path)
    launched = False

    def fake_popen(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("must not launch")

    state = guard.run_guard(
        cwd=cwd,
        expected_commit=expected,
        output_dir=tmp_path / "out",
        lease_dir=tmp_path / "leases",
        gpu_indices=[0],
        poll_seconds=30,
        max_wait_seconds=60,
        child_argv=["python", "train.py"],
        probe=lambda _index: (_ for _ in ()).throw(guard.GuardError("bad nvidia output")),
        popen=fake_popen,
        getpgid=lambda pid: pid,
        killpg=lambda _pgid, _signum: None,
    )

    assert state == "failed_preflight"
    assert not launched


def test_external_pid_interrupts_only_own_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cwd, expected = ready_guard(monkeypatch, tmp_path)
    idle = guard.GpuSnapshot(1, "GPU-test", 0, frozenset())
    contention = guard.GpuSnapshot(1, "GPU-test", 900, frozenset({777}))
    killed: list[tuple[int, int]] = []

    state = guard.run_guard(
        cwd=cwd,
        expected_commit=expected,
        output_dir=tmp_path / "out",
        lease_dir=tmp_path / "leases",
        gpu_indices=[1],
        poll_seconds=30,
        max_wait_seconds=60,
        child_argv=["python", "train.py"],
        probe=snapshots(idle, idle, contention),
        lease_factory=lambda _uuid: FakeLease(),
        popen=lambda *_args, **_kwargs: FakeProcess(456, [None, 143]),
        getpgid=lambda _pid: 777,
        killpg=lambda pgid, signum: killed.append((pgid, signum)),
    )

    assert state == "interrupted_contention"
    assert killed == [(456, guard.signal.SIGTERM)]
    status = json.loads((tmp_path / "out" / "status.json").read_text(encoding="utf-8"))
    assert status["external_compute_pids"] == [777]
    assert status["exit_code"] == 143


def test_lease_contention_waits_without_second_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cwd, expected = ready_guard(monkeypatch, tmp_path)
    idle = guard.GpuSnapshot(2, "GPU-locked", 0, frozenset())
    calls = 0
    ticks = iter([0.0, 0.0, 1.0])

    def fake_probe(_index):
        nonlocal calls
        calls += 1
        return idle

    state = guard.run_guard(
        cwd=cwd,
        expected_commit=expected,
        output_dir=tmp_path / "out",
        lease_dir=tmp_path / "leases",
        gpu_indices=[2],
        poll_seconds=30,
        max_wait_seconds=1,
        child_argv=["python", "train.py"],
        probe=fake_probe,
        lease_factory=lambda _uuid: None,
        monotonic=lambda: next(ticks),
        sleep=lambda _seconds: None,
        getpgid=lambda pid: pid,
        killpg=lambda _pgid, _signum: None,
    )

    assert state == "timed_out"
    assert calls == 1


def test_popen_error_releases_acquired_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cwd, expected = ready_guard(monkeypatch, tmp_path)
    idle = guard.GpuSnapshot(0, "GPU-test", 0, frozenset())
    lease = FakeLease()

    state = guard.run_guard(
        cwd=cwd,
        expected_commit=expected,
        output_dir=tmp_path / "out",
        lease_dir=tmp_path / "leases",
        gpu_indices=[0],
        poll_seconds=30,
        max_wait_seconds=60,
        child_argv=["python", "train.py"],
        probe=snapshots(idle, idle),
        lease_factory=lambda _uuid: lease,
        popen=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot start")),
        getpgid=lambda pid: pid,
        killpg=lambda _pgid, _signum: None,
    )

    assert state == "failed_launch"
    assert lease.released


def test_contention_waits_for_own_exit_after_escalation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cwd, expected = ready_guard(monkeypatch, tmp_path)
    idle = guard.GpuSnapshot(1, "GPU-test", 0, frozenset())
    contention = guard.GpuSnapshot(1, "GPU-test", 100, frozenset({777}))
    events: list[str] = []
    signals: list[int] = []
    lease = FakeLease()
    original_status = guard._status

    class Clock:
        value = 0.0

        def now(self) -> float:
            return self.value

        def sleep(self, seconds: float) -> None:
            self.value += seconds

    clock = Clock()

    def record_status(output_dir, state, **details):
        events.append(state)
        original_status(output_dir, state, **details)

    monkeypatch.setattr(guard, "_status", record_status)
    state = guard.run_guard(
        cwd=cwd,
        expected_commit=expected,
        output_dir=tmp_path / "out",
        lease_dir=tmp_path / "leases",
        gpu_indices=[1],
        poll_seconds=30,
        max_wait_seconds=60,
        child_argv=["python", "train.py"],
        probe=snapshots(idle, idle, contention),
        lease_factory=lambda _uuid: lease,
        popen=lambda *_args, **_kwargs: FakeProcess(456, [None] * 6 + [143]),
        getpgid=lambda _pid: 777,
        killpg=lambda _pgid, signum: signals.append(signum),
        monotonic=clock.now,
        sleep=clock.sleep,
        running_poll_seconds=1,
        terminate_grace_seconds=1,
        kill_grace_seconds=1,
    )

    assert state == "interrupted_contention"
    assert signals == [guard.signal.SIGTERM, guard.KILL_SIGNAL]
    assert "termination_pending" in events
    assert lease.released
    status = json.loads((tmp_path / "out" / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "interrupted_contention"
    assert status["exit_code"] == 143


def test_probe_requests_actual_free_memory_and_keeps_real_compute_pids() -> None:
    commands: list[list[str]] = []
    replies = iter(["3, GPU-shared, 24576, 16384\n", "GPU-shared, 99\nGPU-other, 101\n"])

    def command_runner(command):
        commands.append(list(command))
        return next(replies)

    snapshot = guard.probe_gpu(3, command_runner=command_runner)

    assert "--query-gpu=index,uuid,memory.used,memory.free" in commands[0]
    assert snapshot == guard.GpuSnapshot(3, "GPU-shared", 24576, frozenset({99}), 16384)


@pytest.mark.parametrize("free", ["N/A", "not-a-number", "-1"])
def test_probe_rejects_invalid_free_memory(free: str) -> None:
    replies = iter([f"0, GPU-shared, 24576, {free}\n", "GPU-shared, 99\n"])
    with pytest.raises(guard.GuardError):
        guard.probe_gpu(0, command_runner=lambda _command: next(replies))


@pytest.mark.parametrize("free,threshold", [(None, 16384), (0, 16384), (16383, 16384), (20000, 24576)])
def test_sharing_requires_measured_free_memory_before_acquiring_lease(
    free: int | None, threshold: int,
) -> None:
    snapshot = guard.GpuSnapshot(0, "GPU-shared", 24576, frozenset({99}), free)
    leases_requested = []

    def acquire(uuid):
        leases_requested.append(uuid)
        return FakeLease()

    selected = guard._candidate(
        [0], lambda _index: snapshot, acquire,
        allow_sharing=True, min_free_memory_mib=threshold,
    )

    assert selected is None
    assert leases_requested == []


def test_sharing_accepts_threshold_equality_only_after_lease_and_second_probe() -> None:
    first = guard.GpuSnapshot(0, "GPU-shared", 24000, frozenset({99}), 17000)
    second = guard.GpuSnapshot(0, "GPU-shared", 24616, frozenset({99, 101}), 16384)
    replies = iter([first, second])
    lease = FakeLease()
    events = []

    def probe(index):
        events.append(("probe", index))
        return next(replies)

    def acquire(uuid):
        events.append(("lease", uuid))
        return lease

    selected = guard._candidate(
        [0], probe, acquire, allow_sharing=True, min_free_memory_mib=16384,
    )

    assert selected == (second, lease)
    assert events == [("probe", 0), ("lease", "GPU-shared"), ("probe", 0)]
    assert lease.released is False
    lease.release()


@pytest.mark.parametrize("race", ["free_memory", "uuid"])
def test_sharing_second_probe_race_releases_lease(race: str) -> None:
    first = guard.GpuSnapshot(0, "GPU-shared", 12000, frozenset({99}), 28000)
    second = (
        guard.GpuSnapshot(0, "GPU-shared", 25000, frozenset({99, 101}), 15000)
        if race == "free_memory"
        else guard.GpuSnapshot(0, "GPU-replaced", 12000, frozenset({99}), 28000)
    )
    lease = FakeLease()

    selected = guard._candidate(
        [0], snapshots(first, second), lambda _uuid: lease,
        allow_sharing=True, min_free_memory_mib=16384,
    )

    assert selected is None
    assert lease.released


def test_sharing_leased_second_probe_error_releases_lease() -> None:
    first = guard.GpuSnapshot(0, "GPU-shared", 12000, frozenset({99}), 28000)
    lease = FakeLease()
    count = 0

    def probe(_index):
        nonlocal count
        count += 1
        if count == 1:
            return first
        raise guard.GuardError("snapshot became unreadable")

    with pytest.raises(guard.GuardError, match="unreadable"):
        guard._candidate(
            [0], probe, lambda _uuid: lease,
            allow_sharing=True, min_free_memory_mib=16384,
        )
    assert lease.released


def test_sharing_continues_with_foreign_pids_and_records_actual_observations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cwd, expected = ready_guard(monkeypatch, tmp_path)
    first = guard.GpuSnapshot(0, "GPU-shared", 12000, frozenset({99}), 28000)
    second = guard.GpuSnapshot(0, "GPU-shared", 13000, frozenset({99, 101}), 27000)
    running = guard.GpuSnapshot(0, "GPU-shared", 35000, frozenset({99, 101, 456, 777}), 5000)
    lease = FakeLease()
    killed = []

    state = guard.run_guard(
        cwd=cwd, expected_commit=expected, output_dir=tmp_path / "out",
        lease_dir=tmp_path / "leases", gpu_indices=[0], poll_seconds=30, max_wait_seconds=60,
        child_argv=["python", "train.py"], allow_sharing=True, min_free_memory_mib=16384,
        probe=snapshots(first, second, running), lease_factory=lambda _uuid: lease,
        popen=lambda *_args, **_kwargs: FakeProcess(456, [None, 0]),
        sleep=lambda _seconds: None, getpgid=lambda pid: pid,
        killpg=lambda pgid, signum: killed.append((pgid, signum)),
    )

    assert state == "completed"
    assert killed == []
    assert lease.released
    status = json.loads((tmp_path / "out/status.json").read_text(encoding="utf-8"))
    assert status["allow_sharing"] is True
    assert status["min_free_memory_mib"] == 16384
    assert status["formal_timing_eligible"] is False
    assert status["gpu_memory_free_mib_before_start"] == 27000
    assert status["gpu_compute_pids_before_start"] == [99, 101]
    rows = [json.loads(line) for line in (tmp_path / "out/sharing-observations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 2
    assert any(
        row["memory_used_mib"] == 13000 and row["memory_free_mib"] == 27000
        and row["compute_pids"] == [99, 101] and row["uuid"] == "GPU-shared"
        for row in rows
    )
    assert any(
        row["memory_used_mib"] == 35000 and row["memory_free_mib"] == 5000
        and row["compute_pids"] == [99, 101, 456, 777] and row["uuid"] == "GPU-shared"
        for row in rows
    )


def test_sharing_cancellation_signals_only_own_child_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cwd, expected = ready_guard(monkeypatch, tmp_path)
    shared = guard.GpuSnapshot(0, "GPU-shared", 12000, frozenset({99, 777}), 28000)
    lease = FakeLease()
    killed = []

    def popen(*_args, **_kwargs):
        (tmp_path / "out/cancel").touch()
        return FakeProcess(456, [None, 143])

    state = guard.run_guard(
        cwd=cwd, expected_commit=expected, output_dir=tmp_path / "out",
        lease_dir=tmp_path / "leases", gpu_indices=[0], poll_seconds=30, max_wait_seconds=60,
        child_argv=["python", "train.py"], allow_sharing=True,
        probe=snapshots(shared, shared), lease_factory=lambda _uuid: lease, popen=popen,
        getpgid=lambda pid: pid, killpg=lambda pgid, signum: killed.append((pgid, signum)),
    )

    assert state == "cancelled"
    assert killed == [(456, guard.signal.SIGTERM)]
    assert lease.released
    status = json.loads((tmp_path / "out/status.json").read_text(encoding="utf-8"))
    assert status["state"] == "cancelled"
    assert status["formal_timing_eligible"] is False
    assert status["gpu_compute_pids_before_start"] == [99, 777]


def test_sharing_is_rejected_for_formal_timing_before_any_probe_or_launch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cwd, expected = ready_guard(monkeypatch, tmp_path)
    calls = []

    with pytest.raises(guard.GuardError, match="formal"):
        guard.run_guard(
            cwd=cwd, expected_commit=expected, output_dir=tmp_path / "out",
            lease_dir=tmp_path / "leases", gpu_indices=[0], poll_seconds=30, max_wait_seconds=60,
            child_argv=["python", "benchmark.py"], allow_sharing=True, formal_timing=True,
            probe=lambda index: calls.append(("probe", index)),
            popen=lambda *_args, **_kwargs: calls.append(("popen", None)),
            getpgid=lambda pid: pid, killpg=lambda _pgid, _signum: None,
        )

    assert calls == []


def test_sharing_cli_is_explicit_and_preserves_exclusive_defaults() -> None:
    required = [
        "--cwd", "/checkout", "--expected-commit", "a" * 40,
        "--output-dir", "/out", "--lease-dir", "/leases", "--gpu-index", "0",
        "--poll-seconds", "30", "--max-wait-seconds", "60",
    ]
    default = guard.parse_args([*required, "--", "python", "train.py"])
    assert default.allow_sharing is False
    assert default.formal_timing is False
    assert default.min_free_memory_mib == 16384
    selected = guard.parse_args([
        *required, "--allow-sharing", "--min-free-memory-mib", "20480", "--", "python", "train.py",
    ])
    assert selected.allow_sharing is True
    assert selected.min_free_memory_mib == 20480
    assert selected.child_argv == ["python", "train.py"]


def test_main_forwards_explicit_sharing_and_formal_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def run_guard(**kwargs):
        seen.update(kwargs)
        return "completed"

    monkeypatch.setattr(guard, "run_guard", run_guard)
    result = guard.main([
        "--cwd", "/checkout", "--expected-commit", "a" * 40,
        "--output-dir", "/out", "--lease-dir", "/leases", "--gpu-index", "0",
        "--poll-seconds", "30", "--max-wait-seconds", "60",
        "--allow-sharing", "--min-free-memory-mib", "24576", "--formal-timing",
        "--", "python", "train.py",
    ])
    assert result == 0
    assert seen["allow_sharing"] is True
    assert seen["min_free_memory_mib"] == 24576
    assert seen["formal_timing"] is True

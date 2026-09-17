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
            "0, GPU-test, 42\n1, GPU-other, 900\n",
            "",
        ]
    )

    snapshot = guard.probe_gpu(0, command_runner=lambda _command: next(replies))

    assert snapshot == guard.GpuSnapshot(0, "GPU-test", 42, frozenset())

    bad_replies = iter(["0, GPU-test, 42\n", "GPU-test, not-a-pid\n"])
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
    unsafe = guard.GpuSnapshot(0, "GPU-busy", 12, frozenset({99}))
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

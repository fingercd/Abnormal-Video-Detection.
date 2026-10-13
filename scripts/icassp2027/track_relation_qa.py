"""Run a remote QA under a local ONLINE SwanLab session, without moving secrets.

The remote launcher must be a reviewed, bounded script with __SWANLAB_URL__ as
its tracking URL placeholder. GPU work is launched only after online init.
Its EVENT JSON lines are uploaded live; raw logs remain local on every failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path


def paired_quality_gate(provenance):
    """Check actual paired prediction identities before opening frame truth."""
    c, task = provenance["campaign"], provenance["task"]
    root = Path(c["output_root"])
    dense = (root / (task["dataset"] + "-dense") / "scoring/predictions/resolved.json").as_posix()
    method = (root / task["id"] / "scoring/predictions/resolved.json").as_posix()
    expected = c["datasets"][task["dataset"]]["head_checkpoint"]["sha256"]
    # Standard-library-only remote validation; no model or metric execution.
    code = ("import json\nfrom pathlib import Path\n"
            f"a=json.loads(Path({dense!r}).read_text())\n"
            f"b=json.loads(Path({method!r}).read_text())\n"
            f"if not a['checkpoint_sha256']==b['checkpoint_sha256']=={expected!r}: raise ValueError('paired head SHA differs')\n"
            "if a['evaluation_sampling_fingerprint']!=b['evaluation_sampling_fingerprint']: raise ValueError('paired input/sampling SHA differs')\n"
            "print('PAIRED_QUALITY_IDENTITY_GATE_PASSED',flush=True)\n")
    return ("set -e\n/users/fotile/VAD/.encoder-envs/v2/foundation-video-v2/bin/python - <<'QUALITY_GATE'\n"
            + code + "QUALITY_GATE\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launch-script", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--job-type", default="engineering-qa")
    p.add_argument("--group", default="videomaev2-fit-only")
    p.add_argument("--completion-key", default="qa/completed")
    p.add_argument("--failure-key", default="qa/failed")
    p.add_argument("--formal-quality", action="store_true")
    p.add_argument("--host", default="fotile@121.196.228.153")
    p.add_argument("--port", type=int, default=12345)
    args = p.parse_args()
    launcher = args.launch_script.read_text(encoding="utf-8-sig")
    if launcher.count("__SWANLAB_URL__") != 1:
        raise ValueError("launcher must contain exactly one tracking URL placeholder")
    provenance = json.loads(args.provenance.read_text(encoding="utf-8-sig"))
    # Read at subprocess start so an already-running controller can adopt a
    # verified execution snapshot without touching its active GPU/loggers.
    override_path = args.provenance.parents[2] / "pending-execution.json"
    if provenance.get("phase") == "test" and override_path.exists():
        while True:
            override = json.loads(override_path.read_text(encoding="utf-8"))
            if override["status"] == "ready":
                break
            if override["status"] != "preparing":
                raise RuntimeError("execution optimization held; inspect acceptance receipt")
            print("Waiting for execution-only optimization acceptance; GPU not launched", flush=True)
            time.sleep(10)
        receipt = Path(override["acceptance_receipt"])
        if hashlib.sha256(receipt.read_bytes()).hexdigest() != override["acceptance_sha256"]:
            raise ValueError("execution acceptance receipt SHA mismatch")
        accepted = json.loads(receipt.read_text(encoding="utf-8"))
        from vadbench.workflows.pipeline_acceptance import validate_acceptance
        validate_acceptance(accepted)
        cloud = Path(override["cloud_verification"])
        if hashlib.sha256(cloud.read_bytes()).hexdigest() != override["cloud_verification_sha256"]:
            raise ValueError("benchmark cloud verification SHA differs")
        verified = json.loads(cloud.read_text(encoding="utf-8"))
        if (not verified["verified"] or verified["state"].upper() not in {"SUCCESS", "FINISHED"}
                or verified["completion_summary"].get("qa/completed", {}).get("value") != 1):
            raise ValueError("benchmark cloud success not verified")
        old_snapshot = provenance["campaign"]["snapshot_root"]
        if launcher.count(old_snapshot) != 2:
            raise ValueError("unexpected snapshot references in pending launcher")
        launcher = launcher.replace(old_snapshot, override["snapshot_root"])
        launcher = launcher.replace(" --phase test", " --execution-optimization "
            + shlex.quote(override["remote_execution_config"]) + " --execution-optimization-sha256 "
            + override["execution_config_sha256"] + " --phase test")
        args.source_archive = Path(override["source_archive"])
        if hashlib.sha256(args.source_archive.read_bytes()).hexdigest() != override["source_archive_sha256"]:
            raise ValueError("optimized source archive SHA mismatch")
        provenance["execution_optimization"] = override
        args.provenance.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        args.launch_script.write_text(launcher, encoding="utf-8")
    provenance["tracking_controller_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if provenance.get("phase") == "quality":
        launcher = paired_quality_gate(provenance) + launcher
    polling = provenance.get("phase") == "test"
    if polling:
        marker='\ntail --pid="$pid" -n +1 -f'
        if marker not in launcher:
            raise ValueError("formal launcher missing known detach boundary")
        # The original nohup job is persistent. End only its SSH launcher;
        # receive progress through short read-only connections instead.
        launcher=launcher.split(marker,1)[0]+"\nexit 0\n"
    args.output.mkdir(parents=True, exist_ok=False)
    import swanlab
    from swanlab.sdk.internal.settings import Settings

    run = swanlab.init(
        project="VAD-Key-Attention-20261013", name=args.name, public=False,
        mode="online", log_dir=str(args.output / "swanlog"),
        job_type=args.job_type, group=args.group,
        config={**provenance, "tracking": "local_online_remote_event_stream", "formal_quality": args.formal_quality},
        settings=Settings(interactive=False, probe=Settings.Probe(monitor=False, hardware=False)),
    )
    if not run.url or "http" not in run.url:
        run.finish(state="crashed", error="online URL unavailable; GPU not started")
        raise RuntimeError("online SwanLab run was not established")
    link = {"url": run.url, "path": run.path.strip("/"), "run_id": run.id, "name": run.name}
    (args.output / "swanlab.json").write_text(json.dumps(link, indent=2), encoding="utf-8")
    print("SWANLAB " + json.dumps(link), flush=True)
    for artifact in (args.provenance, args.source_archive, Path(__file__)):
        run.save(artifact.resolve(), base_path=artifact.resolve().parent, policy="now")
    command = ["ssh", "-T", "-p", str(args.port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", args.host, "bash", "-s"]
    completed = False
    returncode = None
    step = 0
    error = None
    last_event = None
    try:
        with (args.output / "remote.log").open("w", encoding="utf-8") as log, \
             (args.output / "events.jsonl").open("w", encoding="utf-8") as events:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
            # Binary stdin prevents Windows TextIOWrapper translating LF to CRLF
            # in the shell program sent to Linux bash.
            process.stdin.write(launcher.replace("__SWANLAB_URL__", shlex.quote(run.url)).encode("utf-8"))
            process.stdin.close()
            for raw_line in process.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if not line.startswith("EVENT "):
                    continue
                event = json.loads(line[6:])
                last_event = event
                events.write(json.dumps(event, ensure_ascii=False) + "\n")
                events.flush()
                step += 1
                metrics = event.get("metrics", {})
                case = event.get("case_id")
                if case:
                    condition = case.split("-", 1)[-1]
                    if "depth" in event:
                        condition = condition.replace(f"-d{event['depth']:02d}", f"-block{event['depth'] + 1:02d}")
                    condition = condition.replace("-k50", "-remove50").replace("-k30", "-remove70")
                    metrics = {f"cases/{condition}/{key}": value for key, value in metrics.items()}
                if metrics:
                    run.log(metrics, step=step)
                if event.get("kind") == "provenance":
                    run.config.update({"remote_provenance": event.get("provenance", event)})
                # Text contains receipts/identities, never pixels or credentials.
                run.log_text(key="audit/events", data=json.dumps(event, ensure_ascii=False), step=step)
                completed = completed or event.get("kind") == "completed"
            returncode = process.wait()
            if polling and returncode == 0:
                from attach_relation_run import poll_original_log
                c=provenance["campaign"]
                snapshot=provenance.get("execution_optimization",{}).get("snapshot_root",c["snapshot_root"])
                for line in poll_original_log(c["control_root"],provenance["task"]["id"],snapshot,last_event):
                    log.write(line);log.flush();print(line,end="",flush=True)
                    if not line.startswith("EVENT "):continue
                    event=json.loads(line[6:]);step+=1
                    events.write(json.dumps(event,ensure_ascii=False)+"\n");events.flush()
                    if event.get("metrics"):run.log(event["metrics"],step=step)
                    run.log_text(key="audit/events",data=json.dumps(event,ensure_ascii=False),step=step)
                    completed=completed or event.get("kind")=="completed"
            if returncode != 0 or not completed:
                raise RuntimeError(f"remote QA incomplete: ssh exit={returncode}, completion_event={completed}; inspect PID before retry")
        run.save((args.output / "remote.log").resolve(), base_path=args.output.resolve(), policy="now")
        run.save((args.output / "events.jsonl").resolve(), base_path=args.output.resolve(), policy="now")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        run.log({args.failure_key: 1}, step=step + 1)
    finally:
        run.finish(state="success" if completed and returncode == 0 and error is None else "crashed", error=error)

    # Read server-side status and the uploaded completion scalar, not merely a URL.
    verification = {**link, "verified": False}
    api = swanlab.Api()
    for attempt in range(3):
        remote = api.run(link["path"])
        summary_key = args.failure_key if error else args.completion_key
        summary = remote.summary(keys=[summary_key])
        verification.update(state=remote.state, completion_summary=summary, attempts=attempt + 1)
        if summary.get(summary_key, {}).get("value") == 1 and remote.state.upper() not in {"RUNNING", "PENDING", ""}:
            verification["verified"] = True
            break
        if attempt < 2:
            time.sleep(5)
    (args.output / "cloud-verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print("CLOUD_VERIFICATION " + json.dumps(verification), flush=True)
    if not verification["verified"]:
        raise RuntimeError("remote QA completed but cloud verification pending; retain local logs for resync")
    if error:
        raise RuntimeError(error)


if __name__ == "__main__":
    main()

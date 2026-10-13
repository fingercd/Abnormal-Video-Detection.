"""Run a remote QA under a local ONLINE SwanLab session, without moving secrets.

The remote launcher must be a reviewed, bounded script with __SWANLAB_URL__ as
its tracking URL placeholder. GPU work is launched only after online init.
Its EVENT JSON lines are uploaded live; raw logs remain local on every failure.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launch-script", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--host", default="fotile@121.196.228.153")
    p.add_argument("--port", type=int, default=12345)
    args = p.parse_args()
    launcher = args.launch_script.read_text(encoding="utf-8-sig")
    if launcher.count("__SWANLAB_URL__") != 1:
        raise ValueError("launcher must contain exactly one tracking URL placeholder")
    provenance = json.loads(args.provenance.read_text(encoding="utf-8-sig"))
    args.output.mkdir(parents=True, exist_ok=False)
    import swanlab
    from swanlab.sdk.internal.settings import Settings

    run = swanlab.init(
        project="VAD-Key-Attention-20261013", name=args.name, public=False,
        mode="online", log_dir=str(args.output / "swanlog"),
        job_type="engineering-qa", group="videomaev2-fit-only",
        config={**provenance, "tracking": "local_online_remote_event_stream", "formal_quality": False},
        settings=Settings(interactive=False, probe=Settings.Probe(monitor=False, hardware=False)),
    )
    if not run.url or "http" not in run.url:
        run.finish(state="crashed", error="online URL unavailable; GPU not started")
        raise RuntimeError("online SwanLab run was not established")
    link = {"url": run.url, "path": run.path.strip("/"), "run_id": run.id, "name": run.name}
    (args.output / "swanlab.json").write_text(json.dumps(link, indent=2), encoding="utf-8")
    print("SWANLAB " + json.dumps(link), flush=True)
    for artifact in (args.provenance, args.source_archive):
        run.save(artifact.resolve(), base_path=artifact.resolve().parent, policy="now")
    command = ["ssh", "-T", "-p", str(args.port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=15", args.host, "bash", "-s"]
    completed = False
    returncode = None
    step = 0
    error = None
    try:
        with (args.output / "remote.log").open("w", encoding="utf-8") as log, \
             (args.output / "events.jsonl").open("w", encoding="utf-8") as events:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            process.stdin.write(launcher.replace("__SWANLAB_URL__", shlex.quote(run.url)))
            process.stdin.close()
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                if not line.startswith("EVENT "):
                    continue
                event = json.loads(line[6:])
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
            if returncode != 0 or not completed:
                raise RuntimeError(f"remote QA incomplete: ssh exit={returncode}, completion_event={completed}; inspect PID before retry")
        run.save((args.output / "remote.log").resolve(), base_path=args.output.resolve(), policy="now")
        run.save((args.output / "events.jsonl").resolve(), base_path=args.output.resolve(), policy="now")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        run.finish(state="success" if completed and returncode == 0 and error is None else "crashed", error=error)

    # Read server-side status and the uploaded completion scalar, not merely a URL.
    verification = {**link, "verified": False}
    api = swanlab.Api()
    for attempt in range(3):
        remote = api.run(link["path"])
        summary_key = "qa/failed" if error else "qa/completed"
        summary = remote.summary(keys=[summary_key]).json()
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

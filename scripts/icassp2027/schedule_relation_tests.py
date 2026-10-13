"""Persistent local controller: one task per GPU 0/1/2/3, then CPU exports.

Each subprocess owns one ONLINE SwanLab run and one durable remote task.
On any failed task stop issuing new work; retain logs/PIDs, never blind-retry.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


GPU_UUIDS = [
    "GPU-1cb1d768-9c5f-f2f4-384d-4f2f7d299781",
    "GPU-6a869044-a866-449a-44c9-6827cfa2efe7",
    "GPU-f070f4be-90a4-bc0c-037f-40b84f3a34dd",
    "GPU-df6a6409-c789-4e3c-4f49-3537e3f4a829",
]


class ExistingTracker:
    """Observe an already-online logger without restarting its remote work."""
    def __init__(self, pid, tracking):
        import psutil
        self.pid = pid
        self.tracking = tracking
        self.process = psutil.Process(pid)
        self.created = self.process.create_time()

    def poll(self):
        import psutil
        try:
            if self.process.is_running() and self.process.create_time() == self.created:
                return None
        except psutil.NoSuchProcess:
            pass
        receipt = self.tracking / "cloud-verification.json"
        if not receipt.exists():
            return 1
        verified = json.loads(receipt.read_text("utf-8"))
        return 0 if verified.get("verified") and verified.get("state") == "FINISHED" else 1


def launcher(c, task, slot, phase, campaign_sha):
    base = c["control_root"]
    name = task["id"] + "-" + phase
    gpu = GPU_UUIDS[slot]
    visibility = gpu if phase == "test" else ""
    check = ""
    if phase == "test":
        # GPU0 may share with the precisely observed foreign PID; never stop it.
        allowed = "1827" if slot == 0 else ""
        check = f'''pids=$(nvidia-smi -i {gpu} --query-compute-apps=pid --format=csv,noheader)
for pid in $pids; do [ "$pid" = "{allowed}" ] || {{ echo "Unexpected process $pid on assigned GPU; no launch"; exit 22; }}; done
free=$(nvidia-smi -i {gpu} --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$free" -ge 12288 ] || {{ echo "Insufficient free memory; no launch"; exit 23; }}
'''
    snapshot = c["snapshot_root"]
    py = "/users/fotile/VAD/.encoder-envs/v2/foundation-video-v2/bin/python"
    return f'''set -euo pipefail
[ "$(hostname)" = ibnode3 ]
BASE={shlex.quote(base)}
NAME={shlex.quote(name)}
mkdir -p "$BASE/logs"
test ! -e "$BASE/logs/$NAME.pid"
{check}
cd /users/fotile/VAD
nohup env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES={shlex.quote(visibility)} PYTHONPATH=/users/fotile/icassp2027-runs/env-hooks/vjepa2-cudnn-off:/users/fotile/VAD/.encoder-envs/v2/overlays/videomaev2:{shlex.quote(snapshot)}/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 HF_HOME=/users/fotile/VAD/.cache-v2/huggingface TORCH_HOME=/users/fotile/VAD/.cache-v2/torch OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 {py} {shlex.quote(snapshot)}/scripts/icassp2027/run_relation_test.py --campaign "$BASE/campaign.json" --campaign-sha256 {campaign_sha} --task-id {shlex.quote(task['id'])} --phase {phase} --tracking-url __SWANLAB_URL__ > "$BASE/logs/$NAME.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$BASE/logs/$NAME.pid"
echo "REMOTE_TASK_PID=$pid GPU_SLOT={slot} TASK=$NAME"
tail --pid="$pid" -n +1 -f "$BASE/logs/$NAME.log" &
tail_pid=$!
set +e
wait "$pid"
status=$?
wait "$tail_pid"
echo "$status" > "$BASE/logs/$NAME.exit"
exit "$status"
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--campaign-sha256", required=True)
    p.add_argument("--source-archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume-controller", type=Path)
    p.add_argument("--adopt-controller", type=Path)
    a = p.parse_args()
    c = json.loads(a.campaign.read_text("utf-8"))
    a.output.mkdir(parents=True, exist_ok=False)
    state = {"controller_pid":os.getpid(), "campaign_sha256":a.campaign_sha256,
             "status":"running", "phase":"test", "tasks":{}, "gpu_slots":[0,1,2,3]}
    def save():
        tmp = a.output / "queue.tmp"
        tmp.write_text(json.dumps(state,indent=2),encoding="utf-8")
        tmp.replace(a.output / "queue.json")
    active = {}
    stopped = False
    save()
    recovered=set()
    if a.adopt_controller is not None:
        import psutil
        previous=json.loads((a.adopt_controller / "queue.json").read_text("utf-8"))
        if previous["campaign_sha256"] != a.campaign_sha256 or previous["phase"] != "test":
            raise ValueError("only matching active test controller can be adopted")
        state["tasks"] = previous["tasks"]
        state["adopted_from"] = str(a.adopt_controller)
        for name,row in state["tasks"].items():
            recovered.add(name.removesuffix("-test"))
            row["tracking_directory"] = str(Path(row.get("tracking_directory", a.adopt_controller/name/"tracking")))
            if row["status"] == "completed":
                continue
            slot=row["slot"]
            tracking=Path(row["tracking_directory"])
            directory=tracking.parent
            if row["status"] == "reattaching_existing_remote_task":
                process=ExistingTracker(row["local_pid"],tracking)
                row["logger_adopted_without_restart"] = True
                log=(a.output/(name+"-adopted.log")).open("w",encoding="utf-8")
                active[slot]=(process,log,name)
                continue
            # A pending tracker has not initialized online or launched GPU.
            # Verify that exact local process and absent tracking directory
            # before replacing only its waiting Python controller.
            if row["status"] not in {"running", "waiting_execution_acceptance"} or tracking.exists():
                raise ValueError("task is not an unlaunched waiting tracker")
            old=psutil.Process(row["local_pid"])
            command=old.cmdline()
            if "scripts/icassp2027/track_relation_qa.py" not in command or str(directory/"provenance.json") not in command:
                raise ValueError("waiting tracker process identity differs")
            old.terminate(); old.wait(timeout=15)
            log=(a.output/(name+"-refreshed.log")).open("w",encoding="utf-8")
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,"PYTHONUTF8":"1","PYTHONPATH":str(Path(__file__).resolve().parents[2]/"src")})
            row.update(local_pid=process.pid,status="waiting_execution_acceptance",tracking_directory=str(tracking))
            active[slot]=(process,log,name)
        save()
    if a.resume_controller is not None:
        previous=json.loads((a.resume_controller / "queue.json").read_text("utf-8"))
        for name,row in previous["tasks"].items():
            task_id=name.removesuffix("-test")
            recovered.add(task_id)
            if row["status"] == "completed":
                state["tasks"][name]=row
                continue
            if row["status"] in {"running","reattaching_existing_remote_task"}:
                import psutil
                tracking=Path(row.get("tracking_directory",a.resume_controller/name/"tracking"))
                if psutil.pid_exists(row["local_pid"]) and (tracking/"swanlab.json").exists():
                    process=ExistingTracker(row["local_pid"],tracking)
                    log=(a.output/(name+"-adopted.log")).open("w",encoding="utf-8")
                    active[row["slot"]]=(process,log,name)
                    state["tasks"][name]={**row,"tracking_directory":str(tracking),"logger_adopted_without_restart":True}
                    continue
            slot=row["slot"]
            recovery=a.output/name
            recovery.mkdir()
            cmd=[sys.executable,"scripts/icassp2027/attach_relation_run.py","--campaign",str(a.campaign),
                 "--task-id",task_id,"--previous-tracking",str(row.get("tracking_directory",a.resume_controller/name/"tracking")),
                 "--output",str(recovery/"tracking")]
            if row.get("swanlab_source"):
                cmd.extend(["--swanlab-source",row["swanlab_source"]])
            if row.get("expected_snapshot"):
                cmd.extend(["--expected-snapshot",row["expected_snapshot"]])
            log=(recovery/"controller.log").open("w",encoding="utf-8")
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"PYTHONUTF8":"1"})
            active[slot]=(process,log,name)
            recovered.add(task_id)
            state["tasks"][name]={"status":"reattaching_existing_remote_task","slot":slot,"local_pid":process.pid,
                "tracking_directory":str(recovery/"tracking"),"expected_snapshot":row.get("expected_snapshot",c["snapshot_root"])}
        state["recovered_from"]=str(a.resume_controller)
        save()
    for phase in ("test", "quality"):
        if stopped:
            break
        state["phase"] = phase
        pending = [t for t in c["tasks"] if phase!="test" or t["id"] not in recovered]
        while pending or active:
            for slot in range(4):
                if slot in active or not pending or stopped:
                    continue
                task = pending.pop(0)
                name = task["id"] + "-" + phase
                run = a.output / name
                run.mkdir()
                script = run / "launch.sh"
                script.write_text(launcher(c,task,slot,phase,a.campaign_sha256),encoding="utf-8")
                meta = run / "provenance.json"
                meta.write_text(json.dumps({"campaign":c,"campaign_sha256":a.campaign_sha256,
                    "task":task,"gpu_slot":slot if phase=="test" else None,"phase":phase},indent=2),encoding="utf-8")
                cmd=[sys.executable, "scripts/icassp2027/track_relation_qa.py", "--launch-script",str(script),
                     "--provenance",str(meta),"--source-archive",str(a.source_archive),
                     "--output",str(run / "tracking"),"--name",name,"--job-type","official-"+phase,
                     "--group",task["dataset"],"--completion-key","test/completed",
                     "--failure-key","test/failed","--formal-quality"]
                log=(run / "controller.log").open("w",encoding="utf-8")
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,"PYTHONUTF8":"1","PYTHONPATH":str(Path(__file__).resolve().parents[2]/"src")})
                active[slot]=(process,log,name)
                state["tasks"][name]={"status":"running","slot":slot,"local_pid":process.pid}
                print(f"START {name} slot={slot} local_pid={process.pid}",flush=True)
                save()
            for slot,(process,log,name) in list(active.items()):
                rc=process.poll()
                if rc is None:
                    continue
                log.close()
                state["tasks"][name].update(status="completed" if rc==0 else "failed",exit_code=rc)
                del active[slot]
                print(f"END {name} slot={slot} exit={rc}",flush=True)
                if rc:
                    stopped=True
                    state["status"]="held_after_failure"
                save()
            if stopped and not active:
                state["unissued_tasks"]=[t["id"] for t in pending]
                break
            time.sleep(10)
    if not stopped:
        state["status"]="completed"
    save()


if __name__=="__main__":
    main()

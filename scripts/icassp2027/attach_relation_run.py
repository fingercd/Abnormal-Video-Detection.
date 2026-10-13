"""Recover online logging of an existing nohup task; never launch GPU work."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign",type=Path,required=True)
    p.add_argument("--task-id",required=True)
    p.add_argument("--previous-tracking",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    c=json.loads(a.campaign.read_text("utf-8"))
    info=json.loads((a.previous_tracking/"swanlab.json").read_text("utf-8"))
    events=a.previous_tracking/"events.jsonl"
    consumed=sum(1 for l in events.open(encoding="utf-8") if l.strip())
    verification=json.loads((a.previous_tracking/"cloud-verification.json").read_text("utf-8"))
    step=verification["completion_summary"]["test/failed"]["step"]
    a.output.mkdir(parents=True,exist_ok=False)
    import swanlab
    from swanlab.sdk.internal.settings import Settings
    run=swanlab.init(project="VAD-Key-Attention-20261013",id=info["run_id"],resume="must",mode="online",
        log_dir=str(a.output/"swanlog"),settings=Settings(interactive=False,probe=Settings.Probe(monitor=False,hardware=False)))
    step+=1
    run.log({"transport/recovered":1,"test/failed":0},step=step)
    run.config.update({"transport_recovery":"reattach original PID and original online run; no GPU task replay"})
    name=a.task_id+"-test"
    base=c["control_root"]
    script=f'''set -eu
test "$(hostname)" = ibnode3
BASE={base}
LOG="$BASE/logs/{name}.log"
pid=$(cat "$BASE/logs/{name}.pid")
if [ -d "/proc/$pid" ]; then
 user=$(ps -p "$pid" -o user=)
 command=$(tr '\\0' ' ' < "/proc/$pid/cmdline")
 [ "$user" = fotile ]
 case "$command" in *"$BASE/snapshot/scripts/icassp2027/run_relation_test.py"*) ;; *) echo "PID identity mismatch";exit 24;; esac
fi
tail --pid="$pid" -n +1 -f "$LOG"
'''
    completed=False;error=None;rc=None
    try:
        with (a.output/"events.jsonl").open("w",encoding="utf-8") as saved:
            for attempt in range(3):
                cmd=["ssh","-T","-p","12345","-o","BatchMode=yes","-o","StrictHostKeyChecking=yes",
                     "-o","ConnectTimeout=15","-o","ServerAliveInterval=30","-o","ServerAliveCountMax=3",
                     "fotile@121.196.228.153","bash","-s"]
                process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                process.stdin.write(script.encode("utf-8"));process.stdin.close()
                ordinal=0
                for raw in process.stdout:
                    line=raw.decode("utf-8",errors="replace");print(line,end="",flush=True)
                    if not line.startswith("EVENT "):continue
                    ordinal+=1
                    if ordinal<=consumed:continue
                    e=json.loads(line[6:]);consumed=ordinal;step+=1
                    saved.write(json.dumps(e,ensure_ascii=False)+"\n");saved.flush()
                    if e.get("metrics"):run.log(e["metrics"],step=step)
                    run.log_text(key="audit/recovered_events",data=json.dumps(e,ensure_ascii=False),step=step)
                    if e["kind"]=="failed":raise RuntimeError(e.get("error","remote computation failed"))
                    completed=completed or e["kind"]=="completed"
                rc=process.wait()
                if completed and rc==0:break
                if attempt<2:time.sleep(6)
            if not completed or rc!=0:
                raise RuntimeError("reattach incomplete; original GPU task was NOT replayed")
        run.save((a.output/"events.jsonl").resolve(),base_path=a.output.resolve(),policy="now")
    except Exception as exc:
        error=str(exc)
    finally:
        run.finish(state="success" if completed and error is None else "crashed",error=error)
    if error:raise RuntimeError(error)
    api=swanlab.Api()
    for attempt in range(3):
        remote=api.run(info["path"])
        summary=remote.summary(keys=["test/completed"])
        if remote.state=="FINISHED" and summary.get("test/completed",{}).get("value")==1:
            (a.output/"cloud-verification.json").write_text(json.dumps({"verified":True,"state":remote.state,"summary":summary}),encoding="utf-8")
            return
        time.sleep(5)
    raise RuntimeError("computation finished; recovered cloud verification pending")


if __name__=="__main__":main()

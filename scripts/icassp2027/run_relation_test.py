"""Execute one frozen relation extraction/scoring or CPU quality task."""
import argparse
import json
import traceback


def emit(kind, **values):
    print("EVENT " + json.dumps({"kind":kind,**values},ensure_ascii=False,allow_nan=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign",required=True)
    p.add_argument("--campaign-sha256",required=True)
    p.add_argument("--task-id",required=True)
    p.add_argument("--phase",choices=["test","quality"],default="test")
    p.add_argument("--tracking-url",required=True)
    a=p.parse_args()
    try:
        from vadbench.paper.relation_testing import load_campaign,run_test,run_quality
        c=load_campaign(a.campaign,a.campaign_sha256)
        task=next(t for t in c["tasks"] if t["id"]==a.task_id)
        c={**c,"campaign_path":a.campaign,"campaign_sha256":a.campaign_sha256}
        emit("provenance",campaign_sha256=a.campaign_sha256,task=task,phase=a.phase,tracking_url=a.tracking_url)
        (run_test if a.phase=="test" else run_quality)(c,task,a.tracking_url,emit)
        return 0
    except Exception as exc:
        traceback.print_exc()
        emit("failed",error=str(exc),metrics={"test/failed":1})
        return 1


if __name__=="__main__":
    raise SystemExit(main())

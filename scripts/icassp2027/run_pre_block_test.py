"""One SHA-frozen VideoMAE/TimeSformer extraction and fixed-head prediction."""
import argparse,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--campaign-sha256',required=True);p.add_argument('--task',required=True);p.add_argument('--tracking-url',required=True);args=p.parse_args()
    from vadbench.paper.pre_block_testing import load_campaign,run_test
    c=load_campaign(args.campaign,args.campaign_sha256);task=next(t for t in c['tasks'] if t['id']==args.task)
    def emit(kind,**fields):print('EVENT '+json.dumps({'kind':kind,**fields},ensure_ascii=False),flush=True)
    run_test(c,task,args.tracking_url,emit)


if __name__=='__main__':main()

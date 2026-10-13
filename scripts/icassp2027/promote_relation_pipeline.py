"""Release pending starts only after fit benchmark AND cloud acceptance.

This local watcher never launches GPU work. Downloads only the completed
benchmark's two small receipts, then atomically publishes execution readiness.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from vadbench.workflows.pipeline_acceptance import validate_acceptance


def validate(acceptance, cloud, config):
    validate_acceptance(acceptance)
    if (not cloud['verified'] or cloud['state'].upper() not in {'FINISHED','SUCCESS'}
            or cloud['completion_summary'].get('qa/completed',{}).get('value') != 1):
        raise ValueError('benchmark cloud success not verified')
    if config != acceptance['configuration']:
        raise ValueError('execution configuration differs from accepted benchmark')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pending',type=Path,required=True)
    p.add_argument('--max-wait-seconds',type=int,default=10800)
    a=p.parse_args()
    start=time.monotonic()
    while True:
        override=json.loads(a.pending.read_text('utf-8'))
        if override['status'] != 'preparing':
            return
        cloud_path=Path(override['cloud_verification'])
        if cloud_path.exists():
            try:
                cloud=json.loads(cloud_path.read_text('utf-8'))
            except json.JSONDecodeError:
                # Tracker publishes this external receipt with write_text;
                # don't mistake a concurrent partial write for failed QA.
                time.sleep(1)
                continue
            if not cloud.get('verified') or cloud.get('state','').upper() not in {'FINISHED','SUCCESS'}:
                raise ValueError('benchmark ended without verified cloud success; leave pending gate closed')
            break
        if time.monotonic()-start > a.max_wait_seconds:
            raise TimeoutError('benchmark/cloud acceptance still pending; no GPU work started by watcher')
        time.sleep(10)
    acceptance_path=Path(override['acceptance_receipt'])
    config_path=acceptance_path.with_name('execution-config.json')
    for remote,local in [(override['remote_acceptance'],acceptance_path),
                         (override['remote_execution_config'],config_path)]:
        subprocess.run(['scp','-P','12345','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
            '-o','ConnectTimeout=15','fotile@121.196.228.153:'+remote,str(local)],check=True,timeout=60)
    acceptance=json.loads(acceptance_path.read_text('utf-8'))
    config=json.loads(config_path.read_text('utf-8'))
    validate(acceptance,cloud,config)
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha(override['source_archive']) != override['source_archive_sha256']:
        raise ValueError('candidate archive SHA differs')
    override.update(status='ready',acceptance_sha256=sha(acceptance_path),
        cloud_verification_sha256=sha(cloud_path),execution_config_sha256=sha(config_path),
        promoter_sha256=sha(__file__),reason='fit RGB/pixels exact; pooled bitwise exact or VisionZip native-repeat bounded; CPU4/GPU2 speedup and cloud success verified')
    temporary=a.pending.with_suffix('.tmp')
    temporary.write_text(json.dumps(override,indent=2),encoding='utf-8');temporary.replace(a.pending)
    print('EXECUTION_OPTIMIZATION_READY '+json.dumps(override),flush=True)


if __name__=='__main__':
    main()

"""CPU-only one-time paired quality for the frozen two-encoder campaign."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from vadbench.checkpoints import sha256_file
from vadbench.features import atomic_write_json
from vadbench.paper.pre_block_testing import load_campaign,bound
from vadbench.workflows.urdmu_quality_export import compare_prediction_runs


def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--campaign-sha256',required=True);p.add_argument('--task',required=True);p.add_argument('--tracking-url',required=True);a=p.parse_args()
 c=load_campaign(a.campaign,a.campaign_sha256);t=next(t for t in c['tasks'] if t['id']==a.task)
 if t['method']=='identity':raise ValueError('no dense-vs-itself bootstrap')
 ds=c['datasets'][t['encoder']][t['dataset']];out=Path(c['output_root'])/t['id'];dense=Path(c['output_root'])/f"{t['encoder']}-{t['dataset']}-dense"
 roots=(dense,out);sources=[a.campaign,Path(__file__)]
 for root in roots:
  task=root/'task.json';status=json.loads(task.read_text())
  if status['status']!='prediction_completed':raise ValueError('both native predictions must be complete')
  resolved=root/'scoring/predictions/resolved.json';r=json.loads(resolved.read_text())
  if r['status']!='completed' or r['phase']!='official_frame' or r['mode']!='direct_insert':raise ValueError('not complete official direct insert')
  if r['checkpoint_sha256']!=ds['head_checkpoint']['sha256']:raise ValueError('actual head SHA differs')
  binding=r['artifacts']['predictions'];expected=root/'scoring/predictions/predictions.jsonl'
  if Path(binding['path'])!=expected or bound(binding)!=expected.resolve():raise ValueError('prediction artifact SHA/path differs')
  source=root/'features/native/resolved.json';source_dict=json.loads(source.read_text())
  sources.extend([task,resolved,source,root/'extraction-contract.json'])
  if root==dense:
   baseline=r;baseline_rep=source_dict['spec']['representation']
  else:
   if r['evaluation_sampling_fingerprint']!=baseline['evaluation_sampling_fingerprint'] or r['test_sequence']!=baseline['test_sequence']:raise ValueError('paired sampling/head execution differs')
   rep=source_dict['spec']['representation']
   if any(rep[key]!=baseline_rep[key] for key in ('backbone','output_dim','precision','position_strategy')):raise ValueError('paired encoder identity differs beyond reducer')
 # Source is not a new scorer: reuse the existing interval/paired bootstrap.
 import vadbench.workflows.urdmu_quality_export as export
 import vadbench.workflows.quality_comparison as engine
 sources.extend([Path(export.__file__),Path(engine.__file__)])
 truth=bound(ds['truth']);preds=[root/'scoring/predictions/predictions.jsonl' for root in roots]
 input_sha={str(x.resolve()):sha256_file(x) for x in [*preds,truth,*sources]}
 target=out/'quality-preview.json'
 if target.exists():
  result=json.loads(target.read_text())
  if result['source_sha256']!=input_sha or result['dataset']!=t['dataset'] or result['metric']['bootstrap_seed']!=c['bootstrap_seed']:raise ValueError('existing preview belongs to different inputs/protocol; never overwrite')
  reused=True
 else:
  result=compare_prediction_runs(dense_predictions=preds[0],method_predictions=preds[1],truth_jsonl=truth,dataset=t['dataset'],source_paths=sources,bootstrap_seed=c['bootstrap_seed'])
  if result['source_sha256']!=input_sha:raise ValueError('source changed during CPU comparison')
  result['tracking_url']=a.tracking_url;result['campaign_task']=t
  atomic_write_json(target,result);reused=False
 metric=result['metric']
 print('EVENT '+json.dumps({'kind':'completed','task_id':t['id'],'quality_path':str(target),'quality_sha256':sha256_file(target),'reused_existing':reused,'quality':result,'metrics':{'quality/completed':1,'quality/dense':metric['dense_value'],'quality/method':metric['method_value'],'quality/delta':metric['delta_method_minus_dense']}}),flush=True)


if __name__=='__main__':main()

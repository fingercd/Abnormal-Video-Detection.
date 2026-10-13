"""Fit-only fixed-input native-vs-native numerical repeatability on GPU2."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from types import SimpleNamespace
from vadbench.paper.evaluation import _make_adapter
from vadbench.data.dense_sampling import DenseSamplingPlan
from vadbench.data.manifest import load_manifest_jsonl
from vadbench.data.video import build_clip_batch,probe_video
from vadbench.data.batches import clean_encoder_batch
from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.relation_deployment import RelationDeployment
from vadbench.workflows.extraction_pipeline import VideoMAEv2CPUPipeline
from vadbench.features import atomic_write_json
from vadbench.checkpoints import sha256_file

def emit(kind,**v):print('EVENT '+json.dumps({'kind':kind,**v}),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--previous-qa',required=True);p.add_argument('--project',required=True);p.add_argument('--output',required=True);p.add_argument('--tracking-url',required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 old=json.loads(Path(a.previous_qa).read_text())['provenance']
 for key in ('manifest','role_lock'):
  assert sha256_file(Path(old[key]['path']))==old[key]['sha256']
 roles=json.loads(Path(old['role_lock']['path']).read_text())['partitions']
 records=load_manifest_jsonl(old['manifest']['path']);paths={v['video_id']:Path(v['path']) for v in old['source_videos']}
 assert all(r.split.value=='train' and roles[r.video_id]=='fit' for r in records)
 for v in old['source_videos']:assert sha256_file(paths[v['video_id']])==v['sha256']
 torch.cuda.set_per_process_memory_fraction(.25,0);torch.manual_seed(0);np.random.seed(0)
 adapter,d=_make_adapter(SimpleNamespace(encoder='videomaev2',project=a.project,device='cuda:0',processor_tensor_type=None))
 bridge=create_observation_bridge('videomaev2',adapter);bridge.model.eval().requires_grad_(False)
 result=[]
 for record in records:
  samples=DenseSamplingPlan(16,2,16).sample(probe_video(paths[record.video_id]).num_frames)[:8]
  batch=clean_encoder_batch(build_clip_batch(paths[record.video_id],record.video_id,[s.clip for s in samples]))
  setup=clean_encoder_batch(build_clip_batch(paths[record.video_id],record.video_id,[samples[0].clip]))
  with torch.inference_mode(),bridge.geometry(setup.frame_indices,setup.valid_mask) as geometry: adapter.encode(setup,train=False)
  deployment=RelationDeployment(bridge,geometry.layout,depth=5,method='visionzip_internal',keep_ratio=.5)
  pipeline=VideoMAEv2CPUPipeline(adapter.encoder,workers=1,temporal_cache=True)
  pixels=pipeline.prepare(batch)
  native_pixels=pipeline.native_prepare([list(c) for c in batch.frames]);assert torch.equal(pixels,native_pixels)
  values=[]
  for variant in ('native','optimized','native','optimized','native','optimized','native','optimized'):
   context=pipeline.prepared_context(pixels) if variant=='optimized' else __import__('contextlib').nullcontext()
   with torch.inference_mode(),context,deployment(batch) as intervention:
    output=adapter.encode(batch,train=False);intervention.validate_execution()
   pooled=output.pooled.float().cpu().numpy().copy();values.append((variant,pooled))
  native=[v for k,v in values if k=='native'];optimized=[v for k,v in values if k=='optimized']
  baseline=max(float(np.max(np.abs(x-y))) for x in native for y in native)
  cross=max(float(np.max(np.abs(x-y))) for x in native for y in optimized)
  relative=max(float(np.linalg.norm(x-y)/np.linalg.norm(x)) for x in native for y in optimized)
  row={'video_id':record.video_id,'native_repeat_max_abs':baseline,'native_vs_optimized_max_abs':cross,'relative_l2_max':relative,'pixel_exact':True,'repeats_per_variant':4}
  result.append(row);emit('repeatability',case=row,metrics={'pipeline/native_repeat_max_abs':baseline,'pipeline/native_optimized_max_abs':cross,'pipeline/native_optimized_relative_l2':relative})
 atomic_write_json(out/'result.json',{'cases':result,'tracking_url':a.tracking_url,'scope':'fit_repeatability_not_detection_quality'})
 emit('completed',metrics={'qa/completed':1})
if __name__=='__main__':main()

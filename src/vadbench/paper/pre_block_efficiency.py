"""Exclusive-GPU pre-block timing; no official test inputs or truth."""
from __future__ import annotations
import argparse,json,os,subprocess,time,gc,hashlib
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from vadbench.checkpoints import sha256_file
from vadbench.features import atomic_write_json
from vadbench.paper.pre_block_testing import load_campaign,bound
from vadbench.paper.urdmu_evaluation import _load_source,_query_chunk_receipt
from vadbench.paper.evaluation import _make_adapter
from vadbench.workflows.extraction import representation_from_verified_encoder
from vadbench.workflows.hf_video_pipeline import HFVideoCPUPipeline
from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.pre_block_deployment import PreBlockDeployment
from vadbench.data.dense_sampling import DenseSamplingPlan,aggregate_interval_scores
from vadbench.data.video import build_clip_batch,OpenCVVideoReader
from vadbench.data.batches import clean_encoder_batch
from vadbench.integrations.detectors.urdmu.backend import build_urdmu
from vadbench.integrations.detectors.urdmu.inference import run_urdmu_query_chunked
from vadbench.engine.train import load_checkpoint


def condition_grid():
 return [{'method':'identity','target_block':None,'keep_ratio':None}]+[{'method':m,'target_block':b,'keep_ratio':k} for m in ('attention_topk','visionzip_internal','uniform') for b in (6,9,12) for k in (.5,.3)]


def gpu_receipt():
 uuid=os.environ['VAD_GPU_UUID'];rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
 own=[line.strip() for line in rows.splitlines() if line.split(',')[0].strip()==uuid]
 pids={int(line.split(',')[1]) for line in own}
 if pids-{os.getpid()}:raise RuntimeError('exclusive GPU has a foreign process: '+repr(own))
 return {'uuid':uuid,'physical_slot':int(os.environ['VAD_GPU_SLOT']),'processes':own,'own_pid':os.getpid(),'exclusive':True}


def setup(c,job):
 ds=c['datasets'][job['encoder']][job['dataset']]
 source=_load_source(SimpleNamespace(trained_run=ds['head_run'],dataset=job['dataset'],phase='official_frame'))
 if source.checkpoint_sha256!=ds['head_checkpoint']['sha256']:raise ValueError('fixed formal head SHA differs')
 adapter,definition=_make_adapter(SimpleNamespace(encoder=job['encoder'],project=c['project'],device='cuda:0',processor_tensor_type=ds['processor_tensor_type']))
 rep=representation_from_verified_encoder(runtime_id=job['encoder'],adapter=adapter,verified_encoder_identity=definition['identity'],preprocessing={'profile':adapter.preprocess_profile},readout={'kind':adapter.pooling},reducer={'name':'identity','calibration':'none'},output_dim=768,precision='float32',position_strategy={'kind':'native'})
 if rep.to_dict()!=source.representation.to_dict():raise ValueError('timing native representation is not exact formal head representation')
 bridge=create_observation_bridge(job['encoder'],adapter);bridge._root.eval().requires_grad_(False)
 dep=None if job['method']=='identity' else PreBlockDeployment(bridge,target_depth=job['target_block']-1,method=job['method'],keep_ratio=job['keep_ratio'])
 plan=DenseSamplingPlan(clip_frames=source.sampling.window['clip_frames'],frame_stride=source.sampling.stride['frame_stride'],window_stride=source.sampling.frame_selection['window_stride'],short_policy=source.sampling.frame_selection['short_video_policy'])
 return ds,source,adapter,bridge,dep,plan


def encode(adapter,dep,batch):
 with nullcontext() if dep is None else dep(batch) as ctx:
  output=adapter.encode(batch,train=False)
  if dep is not None:ctx.validate_execution()
 return output.pooled


def model_group(c,job,subset):
 ds,source,adapter,bridge,dep,plan=setup(c,job)
 selected=next(row for row in subset if row['encoder']==job['encoder'] and row['dataset']==job['dataset'])
 by_id={r.video_id:r for r in source.fit_records};rec=by_id[selected['warmup_video_id']];samples=plan.sample(rec.num_frames)[:job['batch_size']]
 batch=clean_encoder_batch(build_clip_batch(rec.resolve_path(source.metadata['source_training_view']['dataset_root']),rec.video_id,[s.clip for s in samples]))
 inputs,lengths=adapter._prepare_inputs(batch)
 # Native pixels already on GPU. Only decoder/preprocessing/H2D are excluded;
 # native adapter/readout, plugin signal/selection/merge and receipts remain.
 adapter._prepare_inputs=lambda batch:(inputs,lengths)
 wall=[];device=[]
 try:
  with torch.inference_mode():
   for _ in range(20):value=encode(adapter,dep,batch)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   for _ in range(40):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize();t=time.perf_counter();start.record();value=encode(adapter,dep,batch);end.record();end.synchronize()
    wall.append((time.perf_counter()-t)*1000);device.append(start.elapsed_time(end))
   if value.shape!=(job['batch_size'],768) or not torch.isfinite(value).all():raise ValueError('encoder timing output invalid')
  memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
  shape={'before_tokens':1568 if job['encoder']=='videomae' else 1569,'after_tokens':1568 if job['encoder']=='videomae' else 1569} if dep is None else dep.context.receipt()
  return {'scope':'GPU_resident_encoder_adapter_readout_and_plugin_no_decoder_preprocessing_H2D','warmup':20,'samples':40,'wall_ms':wall,'cuda_stream_ms':device,'wall_p50_ms':float(np.median(wall)),'wall_p95_ms':float(np.quantile(wall,.95)),'clips_per_second':job['batch_size']*1000/float(np.median(wall)),'memory':memory,'execution':shape,'head_sha256':source.checkpoint_sha256,'native_representation':rep_dict(source),'input_video_id':rec.video_id,'input_sampling':source.sampling.to_dict(),'CPU_and_storage_exclusive':False}
 finally:
  del adapter._prepare_inputs
  if dep is not None:dep.close()


def rep_dict(source):return source.representation.to_dict()


def pipeline_group(c,job,subset):
 ds,source,adapter,bridge,dep,plan=setup(c,job)
 selected=next(row for row in subset if row['encoder']==job['encoder'] and row['dataset']==job['dataset'])
 by_id={r.video_id:r for r in source.fit_records};root=source.metadata['source_training_view']['dataset_root']
 pipeline=HFVideoCPUPipeline(adapter,workers=2,prefetch=2)
 head,receipt=build_urdmu(768,c['upstream_dir'],mode='Test');loaded=load_checkpoint(source.checkpoint,head,map_location='cpu',strict=True,verify=True)
 if loaded['missing_keys'] or loaded['unexpected_keys']:raise ValueError('head strict load failed')
 head.to('cuda:0').eval();rows=[]
 try:
  with torch.inference_mode():
   rec=by_id[selected['warmup_video_id']];samples=plan.sample(rec.num_frames)[:8]
   batch=clean_encoder_batch(build_clip_batch(rec.resolve_path(root),rec.video_id,[s.clip for s in samples]));inputs,lengths=adapter._prepare_inputs(batch)
   adapter._prepare_inputs=lambda batch:(inputs,lengths)
   try:
    for _ in range(20):encode(adapter,dep,batch)
   finally:del adapter._prepare_inputs
   run_urdmu_query_chunked(head,torch.zeros((1,8,768),device='cuda:0'),query_chunk_size=256,source_receipt=source.metadata['backend'])
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   for video in selected['videos']:
    rec=by_id[video['video_id']];path=rec.resolve_path(root)
    # Validate frozen source bytes outside the timed region; never select by
    # a test score or use frame truth in efficiency measurements.
    if sha256_file(path)!=video['sha256']:raise ValueError('frozen training video SHA differs')
    samples=plan.sample(rec.num_frames);vectors=[];stage_encoder=0.;stage_head=0.
    torch.cuda.synchronize();start=time.perf_counter()
    with OpenCVVideoReader(path) as reader,pipeline.batches(reader,rec.video_id,samples,8) as batches:
     for group,batch,prepared in batches:
      clean=clean_encoder_batch(batch);t=time.perf_counter()
      with prepared:pooled=encode(adapter,dep,clean)
      vectors.append(pooled.detach().cpu());stage_encoder+=time.perf_counter()-t
    features=torch.cat(vectors,dim=0)
    if features.shape!=(len(samples),768) or not torch.isfinite(features).all():raise ValueError('full video feature coverage invalid')
    t=time.perf_counter();result,query=run_urdmu_query_chunked(head,features.unsqueeze(0).to('cuda:0'),query_chunk_size=256,source_receipt=source.metadata['backend'],return_receipt=True)
    _query_chunk_receipt(query,expected_chunk=256,length=len(samples));scores=result['frame'].detach().cpu().numpy()[0].astype(np.float64,copy=False);stage_head=time.perf_counter()-t
    frame=aggregate_interval_scores([s.score_frame_start for s in samples],[s.score_frame_end for s in samples],scores,num_frames=rec.num_frames,reduction='mean')
    if not np.isfinite(frame.scores).all() or len(frame.scores)!=rec.num_frames:raise ValueError('native frame projection invalid')
    # Include the same exact run-boundary projection work as native scoring,
    # but no FeatureStore serialization or result-file writes in this scope.
    boundaries=np.flatnonzero((frame.scores[1:]!=frame.scores[:-1])|(frame.contributors[1:]!=frame.contributors[:-1]))+1
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    rows.append({'video_id':rec.video_id,'clips':len(samples),'num_frames':rec.num_frames,'duration_seconds':rec.num_frames/rec.fps,'wall_seconds':elapsed,'encode_transfer_seconds':stage_encoder,'head_transfer_seconds':stage_head,'frame_intervals':len(boundaries)+1,'frame_scores_sha256':hashlib.sha256(frame.scores.tobytes()).hexdigest(),'video_sha256':video['sha256']})
    print('EVENT '+json.dumps({'kind':'video','job_id':job['id'],'metrics':{'efficiency/videos_completed':len(rows)},'video':rows[-1]}),flush=True)
  source.verify_unchanged()
  total=sum(row['wall_seconds'] for row in rows);clips=sum(row['clips'] for row in rows)
  return {'scope':'complete_raw_video_decode_native_preprocess_H2D_encoder_readout_D2H_fixed_head_exact_query256_frame_projection_no_featurestore_disk_serialization','warmup_encoder_batches':20,'warmup_head_sequences':1,'videos':rows,'video_count':len(rows),'total_wall_seconds':total,'total_clips':clips,'clips_per_second':clips/total,'wall_p50_video_seconds':float(np.median([row['wall_seconds'] for row in rows])),'wall_p95_video_seconds':float(np.quantile([row['wall_seconds'] for row in rows],.95)),'memory':{'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()},'head_sha256':source.checkpoint_sha256,'sampling':source.sampling.to_dict(),'pipeline_identity':pipeline.execution_identity,'execution':None if dep is None else dep.context.receipt(),'CPU_and_storage_exclusive':False,'OS_page_cache_controlled':False,'head_is_offline_noncausal':True}
 finally:
  if dep is not None:dep.close()


def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--campaign-sha256',required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--job',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 c=load_campaign(a.campaign,a.campaign_sha256)
 if sha256_file(a.plan)!=a.plan_sha256:raise ValueError('efficiency plan SHA differs')
 plan=json.loads(a.plan.read_text());job=next(j for j in plan['jobs'] if j['id']==a.job)
 if a.output.exists():raise FileExistsError('preserve prior efficiency attempt')
 before=gpu_receipt();torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);torch.cuda.set_per_process_memory_fraction(.5,0);torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 result=model_group(c,job,plan['training_subsets']) if job['scope']=='encoder' else pipeline_group(c,job,plan['training_subsets'])
 after=gpu_receipt();result.update(job=job,status='completed',pid=os.getpid(),gpu_before=before,gpu_after=after,torch_version=torch.__version__,plan_sha256=a.plan_sha256,campaign_sha256=a.campaign_sha256,formal_test_truth_accessed=False)
 atomic_write_json(a.output,result)
 print('EVENT '+json.dumps({'kind':'group_completed','job_id':job['id'],'result_path':str(a.output),'result_sha256':sha256_file(a.output),'summary':{'scope':job['scope'],'encoder':job['encoder'],'method':job['method'],'batch_size':job['batch_size'],'repeat':job['repeat'],'clips_per_second':result['clips_per_second']},'metrics':{'efficiency/group_completed':1}}),flush=True)


if __name__=='__main__':main()

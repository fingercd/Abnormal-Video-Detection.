"""Frozen two-encoder pre-block extraction/prediction using existing stores."""
from __future__ import annotations
import gc,json,random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from vadbench.checkpoints import sha256_file
from vadbench.data.manifest import load_manifest_jsonl,validate_manifest_pair,write_manifest_jsonl
from vadbench.data.feature_contracts import CompatibilityDeclaration,TrainingIdentity,validate_compatibility
from vadbench.features import atomic_write_json
from vadbench.paper.evaluation import _make_adapter
from vadbench.paper.urdmu_evaluation import URDMUEvaluationRequest,_load_source,run_urdmu_evaluation
from vadbench.workflows.extraction import PooledExtractionSpec,extract_pooled_features,make_sampling_identity,representation_from_verified_encoder
from vadbench.workflows.hf_video_pipeline import HFVideoCPUPipeline
from vadbench.engine.coverage import feature_store_coverage
from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.pre_block_deployment import PreBlockDeployment


def bound(binding):
    p=Path(binding['path']).resolve()
    if sha256_file(p)!=binding['sha256']:raise ValueError('frozen SHA differs: '+str(p))
    return p


def load_campaign(path,expected_sha):
    path=bound({'path':str(path),'sha256':expected_sha});c=json.loads(path.read_text())
    if c['encoders']!=['videomae','timesformer']:raise ValueError('scope may not include completed V2/CLIP')
    expected={(enc,ds,m,b,k) for enc in c['encoders'] for ds in ('ucf_crime','xd_violence')
              for m,b,k in [('identity',None,None)]+[(m,b,k) for m in ('attention_topk','visionzip_internal','uniform') for b in (6,9,12) for k in (.5,.3)]}
    actual={(t['encoder'],t['dataset'],t['method'],t['target_block'],t['keep_ratio']) for t in c['tasks']}
    if len(c['tasks'])!=76 or len(actual)!=76 or actual!=expected:raise ValueError('not exact76-condition grid')
    for name,digest in c['implementation_sha256'].items():bound({'path':str(Path(c['snapshot_root'])/name),'sha256':digest})
    bound(c['method_freeze']);bound(c['freeze_markdown'])
    c['campaign_path']=str(path);c['campaign_sha256']=expected_sha
    return c


def evaluation_request(c,t,features,manifest,contract=None):
    ds=c['datasets'][t['encoder']][t['dataset']]
    return URDMUEvaluationRequest(dataset=t['dataset'],phase='official_frame',mode='direct_insert',trained_run=ds['head_run'],
        feature_store=str(features),evaluation_manifest=str(manifest),upstream_dir=c['upstream_dir'],
        output_root=str(Path(c['output_root'])/t['id']/'scoring'),device='cuda:0',run_id='predictions',
        query_chunk_size=256,max_attention_workspace_bytes=2*1024**3,freeze_path=c['freeze_markdown']['path'],
        audit_report=str(bound(ds['audit'])),method_freeze_contract_path=c['method_freeze']['path'],method_freeze_contract_sha256=c['method_freeze']['sha256'],
        feature_contract_path=None if contract is None else str(contract),feature_contract_sha256=None if contract is None else sha256_file(contract))


def run_test(c,t,tracking_url,emit):
    ds=c['datasets'][t['encoder']][t['dataset']];out=Path(c['output_root'])/t['id'];out.mkdir(parents=True,exist_ok=False)
    atomic_write_json(out/'task.json',{'status':'running','task':t,'tracking_url':tracking_url})
    torch.cuda.set_per_process_memory_fraction(c['cuda_memory_fraction'],0);torch.backends.cudnn.enabled=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(1);random.seed(0);np.random.seed(0);torch.manual_seed(0)
    deployment=None
    try:
        source=_load_source(evaluation_request(c,t,out/'features/native',ds['test_manifest']['path']))
        bound(ds['head_checkpoint'])
        if source.checkpoint_sha256!=ds['head_checkpoint']['sha256'] or source.metadata['run_mode']!='formal' or len(source.fit_records)!=ds['training_videos']:raise ValueError('requires matching complete formal head')
        records=tuple(load_manifest_jsonl(bound(ds['test_manifest'])));validate_manifest_pair(source.fit_records,records)
        if len(records)!=ds['test_videos']:raise ValueError('incomplete sealed test membership')
        adapter,definition=_make_adapter(SimpleNamespace(encoder=t['encoder'],project=c['project'],device='cuda:0',processor_tensor_type=ds['processor_tensor_type']))
        bridge=create_observation_bridge(t['encoder'],adapter);bridge._root.eval().requires_grad_(False)
        reducer={'name':'identity','calibration':'none'}
        if t['method']!='identity':
            deployment=PreBlockDeployment(bridge,target_depth=t['target_block']-1,method=t['method'],keep_ratio=t['keep_ratio']);reducer=deployment.reducer_identity
        representation=representation_from_verified_encoder(runtime_id=t['encoder'],adapter=adapter,verified_encoder_identity=definition['identity'],
            preprocessing={'profile':adapter.preprocess_profile},readout={'kind':adapter.pooling},reducer=reducer,output_dim=768,precision='float32',position_strategy={'kind':'native'})
        atomic_write_json(out/'observed-identity.json',{'evaluation_representation':representation.to_dict(),'training_representation':source.representation.to_dict(),'verified_encoder_identity':definition['identity']})
        compatibility=validate_compatibility(CompatibilityDeclaration(mode='direct_insert',training_representation=source.representation,
            evaluation_representation=representation,training_sampling=source.sampling,evaluation_sampling=source.sampling,baseline_evaluation_sampling=source.sampling,
            training_identity=TrainingIdentity({'kind':'urdmu','checkpoint_role':'dense_reference'},source.fit_manifest_digest,source.metadata['seed'],{'steps':3000,'bags_per_class':64}),sampling_change='none'))
        atomic_write_json(out/'preflight.json',{'compatibility':compatibility,'head_sha256':source.checkpoint_sha256,'training_manifest':str(source.fit_manifest_path),'training_videos':len(source.fit_records),'representation':representation.to_dict()})
        emit('preflight',task_id=t['id'],metrics={'test/compatibility_passed':1},representation=representation.to_dict(),head_sha256=source.checkpoint_sha256)
        frames=source.sampling.window['clip_frames'];stride=source.sampling.stride['frame_stride'];window=source.sampling.frame_selection['window_stride']
        args=dict(sampling_kind='dense',clip_frames=frames,frame_stride=stride,window_stride=window,short_policy=source.sampling.frame_selection['short_video_policy'])
        sampling=make_sampling_identity(records,dataset_root=ds['dataset_root'],**args)
        manifest=out/'test.jsonl';write_manifest_jsonl(records,manifest,dataset_root=ds['dataset_root'],require_files=True)
        pipeline=HFVideoCPUPipeline(adapter,workers=2,prefetch=2)
        def progress(row):emit('progress',task_id=t['id'],metrics={'test/videos_completed':row['videos_completed'],'test/videos_total':row['videos_total'],'test/clips_written':row['clips_written']})
        result=extract_pooled_features(PooledExtractionSpec(runtime_id=t['encoder'],verified_encoder_identity=definition['identity'],representation=representation,sampling=sampling,micro_batch_size=8,**args),
            adapter=adapter,manifest=records,dataset_root=ds['dataset_root'],output_root=out/'features',run_id='native',encode_context_factory=deployment,progress=progress,cpu_pipeline=pipeline)
        if deployment is not None:deployment.close()
        if not result.completed:raise RuntimeError('extraction incomplete: '+str(result.failures))
        if result.records_written!=ds['expected_clips']:raise ValueError('native sampler clip coverage differs from freeze')
        root=Path(result.feature_root);coverage=feature_store_coverage(records,root,result.encoder_fingerprint)
        bindings={key:{'path':str(root/name),'sha256':sha256_file(root/name)} for key,name in (('resolved','resolved.json'),('status','status.json'),('index','index.jsonl'))}
        contract=out/'extraction-contract.json'
        atomic_write_json(contract,{'schema':'icassp2027.official-dense-extraction/v1','status':'ready','dataset':t['dataset'],'data_role':'official-sealed-test','test_only':True,
            'feature_store':{'root':str(root),**bindings},'coverage':coverage,'training_view_contract':ds['training_contract'],'sealed_test_manifest':ds['test_manifest'],
            'training_manifest':{'path':str(manifest),'sha256':sha256_file(manifest)},'campaign_task':t,'tracking_url':tracking_url})
        emit('extracted',task_id=t['id'],metrics={'test/extraction_completed':1,'test/clips_total':result.records_written},extraction=asdict(result))
        deployment=None
        del bridge,adapter,pipeline;gc.collect();torch.cuda.empty_cache()
        scored=run_urdmu_evaluation(evaluation_request(c,t,root,manifest,contract))
        load_campaign(c['campaign_path'],c['campaign_sha256'])
        atomic_write_json(out/'task.json',{'status':'prediction_completed','task':t,'extraction':asdict(result),'scoring':asdict(scored),'tracking_url':tracking_url,'formal_metrics_read':False})
        emit('completed',task_id=t['id'],predictions=scored.predictions,metrics={'test/completed':1,'test/videos_total':len(records),'test/clips_total':result.records_written})
    except Exception as exc:
        if deployment is not None:deployment.close()
        atomic_write_json(out/'task.json',{'status':'failed','task':t,'error':str(exc),'tracking_url':tracking_url});raise

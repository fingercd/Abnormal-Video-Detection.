"""CPU-only real pooled-FeatureStore merge equivalence gate.

Run only beneath an already-online SwanLab tracker. No encoder/head forward,
GPU access, frame truth or quality metrics. Selected original assets stay read-only.
"""
import argparse,hashlib,importlib.metadata,importlib.util,json,os,platform,resource
import shutil,subprocess,sys,time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def emit(kind,**values):
    print('EVENT '+json.dumps(dict(kind=kind,**values)),flush=True)


def tree(root):
    root=Path(root)
    if root.is_symlink():raise ValueError('QA source must not be a symlink')
    result={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('QA source tree contains a symlink')
        if path.is_file():result[path.relative_to(root).as_posix()]={'sha256':sha(path),'size':path.stat().st_size}
    return result


def independent_copy(source,target):
    shutil.copytree(source,target,copy_function=shutil.copy2,symlinks=False)
    source_tree=tree(source)
    if source_tree!=tree(target):raise ValueError('independent copied bytes differ')
    for name in source_tree:
        if os.path.samefile(Path(source)/name,Path(target)/name):
            raise ValueError('QA copy unexpectedly shares a source inode')


def candidate(path,expected_sha):
    if sha(path)!=expected_sha:raise ValueError('merger candidate SHA differs')
    spec=importlib.util.spec_from_file_location('audited_streaming_merge',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.StreamingMergeCandidate()


def runtime_receipt(c):
    from vadbench.workflows import extraction
    import vadbench.features as features
    if Path(sys.executable).resolve()!=Path(c['runtime_interpreter']).resolve():
        raise ValueError('QA must use the original scientific interpreter')
    expected=Path(c['snapshot_root'])/'src/vadbench/workflows/extraction.py'
    if Path(extraction.__file__).resolve()!=expected.resolve():
        raise ValueError('QA must import original frozen extraction, not current worktree')
    if sha(expected)!=c['implementation_sha256']['src/vadbench/workflows/extraction.py']:
        raise ValueError('original frozen extraction SHA differs')
    expected_features=Path(c['snapshot_root'])/'src/vadbench/features.py'
    if Path(features.__file__).resolve()!=expected_features.resolve():
        raise ValueError('QA FeatureStore did not come from frozen source')
    return {'executable':sys.executable,'python':platform.python_version(),
            'extraction_sha256':sha(expected),'features_sha256':sha(expected_features),
            'libraries':{name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','timm')}}


def child(a,c):
    runtime=runtime_receipt(c)
    from vadbench.workflows.extraction import _merge_shards
    merger=_merge_shards if a.child_kind=='baseline' else candidate(a.candidate_file,a.candidate_sha256)
    root=a.child_root;rows=[];start=time.perf_counter()
    for shard in sorted((root/'shards').iterdir()):
        t=time.perf_counter();count=merger(shard,sorted((shard/'blocks').iterdir()))
        rows.append({'video_id':shard.name,'clips':count,'wall_seconds':time.perf_counter()-t,
                     'index_sha256':sha(shard/'index.jsonl')})
    t=time.perf_counter();count=merger(root,sorted((root/'shards').iterdir()));global_wall=time.perf_counter()-t
    result={'status':'passed','kind':a.child_kind,'pid':os.getpid(),'runtime':runtime,
            'per_video':rows,'global_clips':count,'global_wall_seconds':global_wall,
            'total_merge_wall_seconds':time.perf_counter()-start,
            'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'global_index_sha256':sha(root/'index.jsonl'),
            'cache_stats':None if a.child_kind=='baseline' else merger.stats}
    a.child_result.write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--campaign-sha256',required=True)
    p.add_argument('--candidate-file',type=Path,required=True);p.add_argument('--candidate-sha256',required=True)
    p.add_argument('--output',type=Path);p.add_argument('--tracking-url')
    p.add_argument('--child-kind',choices=('baseline','candidate'));p.add_argument('--child-root',type=Path);p.add_argument('--child-result',type=Path)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):
        raise ValueError('QA requires explicitly disabled CUDA_VISIBLE_DEVICES')
    if sha(a.campaign)!=a.campaign_sha256:raise ValueError('scientific campaign SHA differs')
    c=json.loads(a.campaign.read_text());runtime=runtime_receipt(c)
    if a.child_kind:
        if not a.child_root.is_dir() or a.child_result.exists():raise ValueError('invalid child root/result; never overwrite')
        child(a,c);return
    if a.output is None or not a.tracking_url or not a.tracking_url.startswith('https://swanlab.cn/'):
        raise ValueError('new QA output and already-online SwanLab tracking URL required')
    if sha(a.candidate_file)!=a.candidate_sha256:raise ValueError('merger candidate SHA differs')
    task_id='videomae-ucf_crime-attention_topk-preblock6-remove50'
    source_root=(Path(c['output_root'])/task_id).resolve()
    a.output=a.output.resolve()
    if a.output.is_relative_to(source_root) or source_root.is_relative_to(a.output):
        raise ValueError('QA output must not overlap selected original source task')
    a.output.mkdir(parents=True,exist_ok=False)
    if json.loads((source_root/'task.json').read_text())['status']!='prediction_completed':
        raise ValueError('real merger QA requires an originally completed source task')
    feature=source_root/'features/native';resolved=json.loads((feature/'resolved.json').read_text())
    status=json.loads((feature/'status.json').read_text())
    if not status['completed'] or status['status']!='completed':raise ValueError('source extraction was not completed')
    ds=c['datasets']['videomae']['ucf_crime'];manifest=Path(ds['test_manifest']['path'])
    if sha(manifest)!=ds['test_manifest']['sha256']:raise ValueError('source manifest SHA differs')
    from vadbench.data.manifest import load_manifest_jsonl
    from vadbench.data.dense_sampling import DenseSamplingPlan
    from vadbench.features import FeatureStore
    sampling=resolved['spec']['sampling']
    plan=DenseSamplingPlan(clip_frames=sampling['window']['clip_frames'],frame_stride=sampling['stride']['frame_stride'],
                          window_stride=sampling['frame_selection']['window_stride'],short_policy=sampling['frame_selection']['short_video_policy'])
    records=tuple(load_manifest_jsonl(manifest));selected=[]
    for label in (False,True):
        for record in sorted((r for r in records if r.is_anomaly==label),key=lambda r:r.video_id):
            expected=len(plan.sample(record.num_frames))
            shard=feature/'shards'/record.video_id
            if expected<64 or not (shard/'index.jsonl').is_file():continue
            if not (shard/'blobs').is_dir():raise ValueError('original published shard blob directory absent; no write allowed')
            count=sum(1 for _ in FeatureStore(shard).iter_records())
            if count!=expected:raise ValueError('completed source video clip coverage differs')
            selected.append({'video_id':record.video_id,'is_anomaly':label,'clips':count,'source':str(shard)});break
        else:raise ValueError('requires complete normal and anomalous videos with >=64 clips')
    before={row['video_id']:tree(row['source']) for row in selected}
    emit('started',tracking_url=a.tracking_url,CPU_only=True,model_loaded=False,test_frame_truth_accessed=False,
         source_task=task_id,selected=selected,metrics={'optimization/merge_gate_started':1})
    copied=a.output/'copied-source';(copied/'shards').mkdir(parents=True)
    for row in selected:independent_copy(Path(row['source']),copied/'shards'/row['video_id'])
    results={}
    for kind in ('baseline','candidate'):
        target=a.output/kind;independent_copy(copied,target)
        result_path=a.output/(kind+'-result.json')
        command=[sys.executable,str(Path(__file__).resolve()),'--campaign',str(a.campaign),'--campaign-sha256',a.campaign_sha256,
                 '--candidate-file',str(a.candidate_file),'--candidate-sha256',a.candidate_sha256,
                 '--child-kind',kind,'--child-root',str(target),'--child-result',str(result_path)]
        launch=time.perf_counter()
        with (a.output/(kind+'-child.log')).open('wb') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode!=0:raise RuntimeError(kind+' child failed; retain QA logs, never alter source')
        result=json.loads(result_path.read_text());result['process_wall_seconds']=time.perf_counter()-launch
        result['command']=command;results[kind]=result
        emit('child_completed',result=result,metrics={'optimization/merge_'+kind+'_wall_seconds':result['total_merge_wall_seconds'],
                                                    'optimization/merge_'+kind+'_peak_RSS_KiB':result['peak_RSS_KiB']})
    baseline=a.output/'baseline';optimized=a.output/'candidate'
    if results['baseline']['global_index_sha256']!=results['candidate']['global_index_sha256']:
        raise ValueError('baseline/candidate global index bytes SHA differ')
    for row in selected:
        if sha(baseline/'shards'/row['video_id']/'index.jsonl')!=sha(optimized/'shards'/row['video_id']/'index.jsonl'):
            raise ValueError('baseline/candidate per-video index bytes SHA differ')
    import numpy as np
    left=FeatureStore(baseline);right=FeatureStore(optimized);verified=0
    for arow,brow in zip(left.iter_records(),right.iter_records(),strict=True):
        if arow.to_dict()!=brow.to_dict():raise ValueError('baseline/candidate rows differ')
        aa=left.load_bundle(arow);bb=right.load_bundle(brow)
        if set(aa)!={'features','pooled'} or set(bb)!=set(aa):raise ValueError('not pooled-only bundles')
        for name in aa:
            ref=arow.arrays[name]
            if (aa[name].shape!=ref.shape or aa[name].dtype.str!=ref.dtype or aa[name].nbytes!=ref.nbytes
                    or not np.isfinite(aa[name]).all() or not np.array_equal(aa[name],bb[name])):
                raise ValueError('real source array/layout/finite equivalence differs')
        verified+=1
    expected=sum(row['clips'] for row in selected)
    stats=results['candidate']['cache_stats']
    if verified!=expected or stats['full_bundle_validations']!=expected or stats['fresh_hash_cache_reuses']!=expected:
        raise ValueError('gate did not exercise full first validation and fresh global content verification')
    after={row['video_id']:tree(row['source']) for row in selected}
    if before!=after:raise ValueError('selected original source assets changed')
    result={'status':'passed','tracking_url':a.tracking_url,'CPU_only':True,'encoder_model_loaded':False,
            'head_model_loaded':False,'test_frame_truth_accessed':False,'quality_metrics_computed':False,
            'role':'real_frozen_FeatureStore_execution_data_integrity_not_quality_or_GPU_efficiency',
            'campaign_sha256':a.campaign_sha256,'script_sha256':sha(__file__),
            'candidate_sha256':a.candidate_sha256,'source_task':task_id,'selected':selected,
            'selected_original_tree_before':before,'selected_original_tree_after':after,
            'source_unchanged':True,'independent_copies':True,'verified_clips':verified,'runtime':runtime,
            'baseline':results['baseline'],'candidate':results['candidate'],
            'index_bytes_exact':True,'arrays_exact':True,
            'total_merge_speedup':results['baseline']['total_merge_wall_seconds']/results['candidate']['total_merge_wall_seconds'],
            'timing_caveat':'CPU/storage shared; baseline first then candidate; OS page cache uncontrolled; two complete selected videos only'}
    report=a.output/'acceptance.json';report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    emit('completed',result_path=str(report),result_sha256=sha(report),
         metrics={'optimization/merge_gate_completed':1,'qa/completed':1,
                  'optimization/merge_gate_speedup':result['total_merge_speedup']})


if __name__=='__main__':main()

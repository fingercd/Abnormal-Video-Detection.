"""One sealed DSANet task, one main paired CI; reuse dense supplementary data."""
from __future__ import annotations
import hashlib
import importlib
import json
import sys
from pathlib import Path
import numpy as np
from vadbench.workflows import quality_comparison as quality
from vadbench.workflows.urdmu_quality_export import _aligned_video, _prediction_rows, _truth_rows
from ._quality_runtime import convert


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paired_identity(campaign,task):
    """Check actual sealed heads, sampling, features and score SHA before truth."""
    ds=campaign['datasets'][task['dataset']]
    root=Path(campaign['output_root'])
    identities=[]
    for name in (task['dataset']+'-dense',task['task_id']):
        run=root/name;completed=read(run/'prediction-completed.json');summary=read(run/'scores/summary.json')
        if completed['status']!='completed' or summary['status']!='completed' or summary['videos']!=ds['videos']:
            raise ValueError('paired scoring coverage incomplete')
        if summary['checkpoint_sha256']!=ds['checkpoint']['sha256'] or summary['manifest_sha256']!=ds['manifest']['sha256']:
            raise ValueError('paired actual head or manifest SHA differs')
        if sha(run/'scores/predictions.jsonl')!=completed['score_sha256'] or sha(run/'scores/summary.json')!=completed['score_summary_sha256']:
            raise ValueError('sealed score artifacts changed')
        identity=read(run/'features/resolved.json')
        if identity['manifest_sha256']!=ds['manifest']['sha256'] or identity['weights_sha256']!=campaign['clip_weights']['sha256']:
            raise ValueError('actual feature manifest/CLIP weights differs')
        identities.append(identity)
    for key in ('manifest_sha256','bridge_sha256','weights_sha256','batch_size','workers','source_group','crop','dtype'):
        if identities[0][key]!=identities[1][key]:raise ValueError('paired extraction identity differs: '+key)
    method=identities[1]
    if method.get('method')!=task['method'] or method.get('block_index')!=task['block']-1 or method['keep_ratio']!=task['keep_ratio']:
        raise ValueError('actual compressed method/layer/budget differs')
    for key,path in (('bridge_sha256','src/vadbench/token_reduction/bridges/clip.py'),('intervention_sha256','src/vadbench/token_reduction/bridges/clip_post_attention.py'),('selector_sha256','src/vadbench/token_reduction/relation_selection.py')):
        if method[key]!=campaign['source_sha256'][path]:raise ValueError('actual implementation SHA differs')
    if sha(ds['manifest']['path'])!=ds['manifest']['sha256'] or sha(ds['truth']['path'])!=ds['truth']['sha256']:
        raise ValueError('frozen manifest or truth file changed')
    return identities


def point_metrics(videos,which):
    """The existing exact weighted engines, with one point and no bootstrap."""
    lengths=[];labels=[];scores=[]
    for video in videos:
        n,l,s=quality._validated_video(video);lengths.append(n);labels.append(l);scores.append(s[which])
    h=quality._histogram(np.concatenate(scores),np.concatenate(lengths),np.concatenate(labels))
    curve=quality._PRCurve.prepare([h]);w=np.ones((1,1),dtype=np.int32)
    roc=float(quality._auc_pair_matrix([h])[0,0]/(h.positive.sum()*h.negative.sum()))
    return {'frame_roc_auc':roc,'frame_ap':float(curve.evaluate(w,integration='step')[0]),
            'frame_pr_auc':float(curve.evaluate(w,integration='trapezoid')[0]),'CI':'not_computed_supplementary_point_metrics'}


def localization(dataset,scores,manifest,upstream):
    """Reuse the author's five tIoU thresholds and ordering, no new thresholds."""
    sys.path.insert(0,str(upstream))
    module=importlib.import_module(f'utils.{dataset}_detectionMAP')
    lists=upstream.parent/'list'
    segments=lists/('gt_segment_ucf.npy' if dataset=='ucf' else 'gt_segment.npy')
    labels=lists/('gt_label_ucf.npy' if dataset=='ucf' else 'gt_label.npy')
    source=[json.loads(s) for s in Path(manifest).read_text().splitlines()]
    rows=[json.loads(s) for s in Path(scores).read_text().splitlines()]
    if [r['video_id'] for r in rows]!=[r['video_id'] for r in source]:raise ValueError('author localization order differs')
    predictions=[]
    for row in rows:
        values=np.asarray(row['class_scores'],dtype=np.float64)
        if values.shape!=(row['feature_rows'],14 if dataset=='ucf' else 7) or not np.isfinite(values).all():raise ValueError('class scores invalid')
        predictions.append(np.repeat(values,16,axis=0))
    values,ious=module.getDetectionMAP(predictions,np.load(segments,allow_pickle=True),np.load(labels,allow_pickle=True),excludeNormal=False)
    if len(values)!=5 or len(ious)!=5:raise ValueError('author map lacks five tIoU thresholds')
    return {'map_at_tiou_percent':{str(float(i)):float(v) for i,v in zip(ious,values)},'mean_map_percent':float(np.mean(values)),
            'segments_sha256':sha(segments),'labels_sha256':sha(labels),'upstream_module_sha256':sha(module.__file__),
            'definition':'original DSANet getDetectionMAP; author order; scores repeated16x; excludeNormal=False'}


def run(options):
    campaign=read(options.campaign)
    task=next(t for t in campaign['tasks'] if t['task_id']==options.task)
    if task['method']=='identity':raise ValueError('quality-one compares a compressed configuration, not dense with itself')
    identities=paired_identity(campaign,task)
    ds=campaign['datasets'][task['dataset']];root=Path(campaign['output_root'])
    dense=root/(task['dataset']+'-dense')/'scores/predictions.jsonl'
    method=root/task['task_id']/'scores/predictions.jsonl'
    sources={'campaign':sha(options.campaign),'truth':ds['truth']['sha256'],'dense_scores':sha(dense),'method_scores':sha(method)}
    # Never overwrite/recompute an existing preview. The controller reuses it.
    options.output.mkdir(parents=True,exist_ok=False)
    if campaign['bootstrap_draws']!=quality.BOOTSTRAP_DRAWS:
        raise ValueError('frozen bootstrap draw count differs from comparison engine')
    truth=_truth_rows(Path(ds['truth']['path']))
    if len(truth)!=ds['videos']:raise ValueError('incomplete canonical truth')
    pairs={};conversion={}
    for branch in ('coarse','refined'):
        aligned={}
        for name,source in (('dense',dense),('method',method)):
            dest=options.output/f'{branch}-{name}-intervals.jsonl'
            conversion[f'{branch}-{name}']=convert(task['dataset'],branch,source,truth,dest)
            aligned[name]=_prediction_rows(dest)
        pairs[branch]=[_aligned_video(i,truth[i],aligned['dense'][i],aligned['method'][i]) for i in sorted(truth)]
    cache_key={'dataset':task['dataset'],'truth_sha256':sources['truth'],'dense_scores_sha256':sources['dense_scores'],
               'manifest_sha256':ds['manifest']['sha256'],'quality_engine_sha256':sha(quality.__file__),
               'single_quality_sha256':sha(__file__),
               'upstream_src':campaign['dsanet_src']}
    cache_path=options.dense_cache/f'{task["dataset"]}-dense-supplementary.json'
    options.dense_cache.mkdir(parents=True,exist_ok=True)
    if cache_path.exists():
        cached=read(cache_path)
        if cached['key']!=cache_key:raise ValueError('dense supplementary cache identity differs')
    else:
        cached={'key':cache_key,'point_metrics':{b:point_metrics(v,0) for b,v in pairs.items()},
                'localization':localization(task['dataset'],dense,ds['manifest']['path'],Path(campaign['dsanet_src']))}
        cache_path.write_text(json.dumps(cached,indent=2),encoding='utf-8')
    metric=campaign['primary_metrics'][task['dataset']]
    result=quality.compare_paired_quality(pairs['coarse'],metric=metric,source_sha256=sources,bootstrap_seed=campaign['bootstrap_seed'])
    # Preserve the expensive one-time CI even if later supplementary export
    # or transport fails. Recovery must reuse this result, not replay bootstrap.
    (options.output/'main-ci.json').write_text(json.dumps({'source_sha256':sources,'metric':result},indent=2),encoding='utf-8')
    points={'dense':cached['point_metrics'],'method':{b:point_metrics(v,1) for b,v in pairs.items()}}
    for which,key in (('dense','dense_value'),('method','method_value')):
        if not np.isclose(points[which]['coarse'][metric],result[key],rtol=0,atol=1e-12):raise ValueError('main and point metric disagree')
    method_localization=localization(task['dataset'],method,ds['manifest']['path'],Path(campaign['dsanet_src']))
    for key in ('segments_sha256','labels_sha256','upstream_module_sha256'):
        if cached['localization'][key]!=method_localization[key]:raise ValueError('paired localization truth/code changed: '+key)
    payload={'status':'completed','task':task,'primary_branch':'coarse','primary_metric':metric,'metric':result,
             'point_metrics':points,'localization':{'dense':cached['localization'],'method':method_localization},
             'actual_feature_identities':identities,'conversion':conversion,'source_sha256':sources,
             'dense_cache_sha256':sha(cache_path),'head_training_seed':'unknown_author_published',
             'bootstrap_seed':campaign['bootstrap_seed'],'bootstrap_draws':campaign['bootstrap_draws'],
             'multiple_configuration_correction':False,'implementation_sha256':sha(__file__)}
    output=options.output/'quality-preview.json';output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('EVENT '+json.dumps({'kind':'completed','task_id':options.task,'quality':payload,
                              'metrics':{'quality/completed':1,'quality/main_method':result['method_value'],'quality/main_dense':result['dense_value'],
                                         'quality/main_delta':result['delta_method_minus_dense'],'quality/ci_low':result['ci_low'],'quality/ci_high':result['ci_high']}}),flush=True)

"""Synthetic contract/copy tests, no pretrained models or real data."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, sys
import pytest
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.cwd() / 'tests/icassp2027'))
from preblock_resume import PreBlockResumeSource, validate_execution_proof
from vadbench.token_reduction.pre_block_deployment import PreBlockDeployment
from vadbench.features import compute_encoder_fingerprint, FeatureStore
from vadbench.workflows.extraction import _semantic_runtime_identity, _samples
from test_feature_resume import _setup, _run, _snapshot


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def proof(tmp_path):
    old = dict(kind='hf_video_ordered_cpu_pipeline_v1', workers=2, prefetch=2,
               implementation_sha256='original-pipeline', scheduler_sha256='scheduler',
               continuous_reader=True, deduplicate_frames=False, processed_frame_tail_cache=False)
    candidate = tmp_path / 'candidate.py'; candidate.write_text('# synthetic candidate')
    new = dict(old, kind='hf_video_ordered_cpu_pipeline_numpy_packing_candidate_r01',
               packing_implementation_sha256=binding(candidate)['sha256'], native_processor_return_tensors='np',
               tensor_conversion='torch.from_numpy', adapter_processor_tensor_type='pt')
    cases = [dict(video_id=str(i), is_anomaly=bool(i % 2), requested_batch_size=b,
                  RGB_exact=True, indices_exact=True, processor_exact=True,
                  inputs={'pixel_values': {'torch_equal': True}}, two_worker_repeated_exact=True)
             for i in range(4) for b in (1, 8)]
    receipt = dict(tracking_url='https://synthetic.example/run',status='passed', CPU_only=True, encoder_model_loaded=False,
                   test_truth_accessed=False, campaign_sha256='campaign', head_sha256='head',
                   candidate_sha256=binding(candidate)['sha256'], frozen_pipeline_sha256=old['implementation_sha256'], cases=cases)
    acceptance = tmp_path / 'acceptance.json'; acceptance.write_text(json.dumps(receipt))
    cloud = tmp_path / 'cloud.json'; cloud.write_text(json.dumps(dict(url='https://synthetic.example/run',verified=True, state='FINISHED', completion_summary={'qa/completed': {'value': 1}})))
    events=tmp_path/'events.jsonl';events.write_text(json.dumps(dict(kind='completed',result_sha256=binding(acceptance)['sha256'],metrics={'qa/completed':1}))+'\n')
    return dict(events=binding(events),acceptance=binding(acceptance), cloud_verification=binding(cloud), candidate=binding(candidate),
                campaign_sha256='campaign', head_sha256='head', old_pipeline=old, new_pipeline=new)


def reduction(r, batch, clip_index=0):
    target=r['target_block']-1; signal=target-1; before=r['source_tokens']; after=r['output_tokens']
    return dict(encoder='videomae', site='previous_complete_block_output_before_target_LN_attention',
                target_block=r['target_block'], signal_block=r['signal_block'], method=r['name'],
                keep_ratio=r['patch_keep_ratio'], before_tokens=before, after_tokens=after,
                inputs={str(i):[batch,before if i<target else after,4] for i in range(12)},
                outputs={str(i):[batch,before if i<signal else after,4] for i in range(12)},
                mlp_inputs={str(i):[batch,before if i<target else after,4] for i in range(12)},
                budget_unit='tubelet_patch', time_frames_preserved=None, spatial_packing=None,
                native_attention_backend_unchanged=True, forward_count=clip_index//8+1,
                gathered_tokens=after, actual_patch_keep_ratio=after/before,
                source_membership_available=True, mass_used_to_reweight_readout=False)


@pytest.mark.parametrize('ratio', [.5,.3])
def test_complete_copy_partial_refused_and_source_bytes_unchanged(tmp_path, ratio):
    records, adapter, spec = _setup(tmp_path, 'dense')
    initial = _run(tmp_path, records, adapter, spec, 'initial')
    assert initial.completed
    source = Path(initial.run_dir); resolved=json.loads((source/'resolved.json').read_text())
    pr=proof(tmp_path)
    r=dict(name='attention_topk', encoder_id='videomae', target_block=9, signal_block=8,
           patch_keep_ratio=ratio, source_tokens=1568, output_tokens=784 if ratio==.5 else 470)
    resolved['spec']['runtime_id']='videomae';resolved['spec']['micro_batch_size']=8
    resolved['spec']['representation']['reducer']=r
    resolved['spec']['cpu_pipeline']=pr['old_pipeline']
    resolved['runtime']['runtime_id']='videomae'
    resolved['encoder_fingerprint']=compute_encoder_fingerprint({'paper_pooled_cache':_semantic_runtime_identity(resolved['runtime'])})
    (source/'resolved.json').write_text(json.dumps(resolved))
    source_sha=binding(source/'resolved.json')['sha256']
    for rec in records:
        index=source/'shards'/rec.video_id/'index.jsonl'
        rows=[json.loads(l) for l in index.read_text().splitlines()]
        count=len(rows)
        for row in rows:
            row['encoder_fingerprint']=resolved['encoder_fingerprint']
            m=row['metadata'];m['runtime_reference']['sha256']=source_sha;m['runtime']['runtime_id']='videomae'
            m['encoder_output']['token_count']=r['output_tokens']
            m['reduction_execution']=reduction(r,min(8,count-(row['clip_index']//8)*8),row['clip_index'])
        if rec.video_id=='two':rows=rows[:-1] # valid but incomplete published legacy index
        index.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    before=_snapshot(source)
    target=tmp_path/'resumed';target.mkdir()
    current=deepcopy(resolved);current['spec']['cpu_pipeline']=pr['new_pipeline']
    (target/'resolved.json').write_text(json.dumps(current))
    # Actual type gate exercised; this is a synthetic deployment instance only.
    dep=object.__new__(PreBlockDeployment);dep._identity=r;dep.context=SimpleNamespace(blocks=[None]*12)
    reuse=PreBlockResumeSource(source,target,current,encode_context_factory=dep,execution_proof=pr)
    for rec in records:
        oldrow=FeatureStore(source/'shards'/rec.video_id).records()[0]
        count=reuse.copy_video(record=rec,samples=_samples(spec,64),source_video=oldrow.metadata['source_video'],
              destination=target/'shards'/rec.video_id,runtime_reference={'base':'extraction_run','path':'resolved.json','sha256':binding(target/'resolved.json')['sha256']})
        if rec.video_id=='two':assert count==0 and not (target/'shards'/rec.video_id).exists()
        else:
            assert count==31
            newrows=FeatureStore(target/'shards'/rec.video_id).records()
            assert all(row.metadata['reused_from']['source_run']==str(source) for row in newrows)
            assert all(row.metadata['reduction_execution']==FeatureStore(source/'shards'/rec.video_id).records()[i].metadata['reduction_execution'] for i,row in enumerate(newrows))
    assert before==_snapshot(source)
    bad=deepcopy(current);bad['spec']['micro_batch_size']=4
    with pytest.raises(ValueError,match='spec differs'):
        PreBlockResumeSource(source,tmp_path/'bad',bad,encode_context_factory=dep,execution_proof=pr)


@pytest.mark.parametrize('tamper',['candidate','workers','cloud','cases'])
def test_execution_gate_tamper_rejected(tmp_path,tamper):
    p=proof(tmp_path);old=p['old_pipeline'];new=p['new_pipeline']
    if tamper=='candidate':Path(p['candidate']['path']).write_text('changed')
    elif tamper=='workers':new['workers']=4
    elif tamper=='cloud':
        q=Path(p['cloud_verification']['path']);v=json.loads(q.read_text());v['state']='RUNNING';q.write_text(json.dumps(v));p['cloud_verification']=binding(q)
    else:
        q=Path(p['acceptance']['path']);v=json.loads(q.read_text());v['cases'][0]['inputs']['pixel_values']['torch_equal']=False;q.write_text(json.dumps(v));p['acceptance']=binding(q)
    with pytest.raises(ValueError):validate_execution_proof(p,old,new)


@pytest.mark.parametrize('field',['signal_block','target_block','after_tokens','inputs','forward_count'])
def test_actual_preblock_receipt_tamper_rejected(field):
    reuse=object.__new__(PreBlockResumeSource);reuse.reducer=dict(name='attention_topk',target_block=9,signal_block=8,patch_keep_ratio=.3,source_tokens=1568,output_tokens=470);reuse.depth=12;reuse.dim=4
    row=reduction(reuse.reducer,7)
    reuse._validate_reduction(row,clip_index=24,clip_count=31,token_count=470)
    row[field]=0
    with pytest.raises(ValueError):reuse._validate_reduction(row,clip_index=24,clip_count=31,token_count=470)


@pytest.mark.parametrize('kind', ['packing', 'dedup'])
def test_both_explicit_execution_schemas_accepted(tmp_path,kind):
    p=proof(tmp_path)
    if kind=='dedup':
        p['new_pipeline'].update(kind='hf_video_batch_local_frame_dedup_candidate_r02',
            deduplicate_frames=True,dedup_scope='single_video_single_prepare_call',
            dedup_duplicate_rgb_exact_check=True,processed_frame_tail_cache=False,
            native_processor_unique_group_frames=16)
    validate_execution_proof(p,p['old_pipeline'],p['new_pipeline'])


@pytest.mark.parametrize('tamper', ['eventsha','cloudurl','scheduler','extra'])
def test_event_identity_and_exact_pipeline_schema_strict(tmp_path,tamper):
    p=proof(tmp_path)
    if tamper=='eventsha':
        q=Path(p['events']['path']);v=json.loads(q.read_text());v['result_sha256']='changed';q.write_text(json.dumps(v));p['events']=binding(q)
    elif tamper=='cloudurl':
        q=Path(p['cloud_verification']['path']);v=json.loads(q.read_text());v['url']='different';q.write_text(json.dumps(v));p['cloud_verification']=binding(q)
    elif tamper=='scheduler':p['new_pipeline']['scheduler_sha256']='different'
    else:p['new_pipeline']['unvalidated_optimization']=True
    with pytest.raises(ValueError):validate_execution_proof(p,p['old_pipeline'],p['new_pipeline'])

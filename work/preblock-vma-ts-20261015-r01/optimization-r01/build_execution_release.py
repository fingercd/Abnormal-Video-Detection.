"""Freeze actual execution sources/proofs without changing scientific campaign."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tarfile

r = Path(__file__).resolve().parent.parent
o = r / 'optimization-r01'
repo = r.parents[1]
sys.path.insert(0, str(repo / 'src'))
sys.path.insert(0, str(o))
from preblock_resume import validate_execution_proof


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding(p):
    return {'path': str(Path(p).resolve()), 'sha256': sha(p)}


c = json.loads((r / 'campaign-r02.json').read_text())
scientific_sha = sha(r / 'campaign-r02.json')
stage = o / 'execution-release-r03'
stage.mkdir(exist_ok=False)
remote = c['control_root'] + '/execution-optimization-r03'
copies = {
    'numpy_dedup_candidate.py': o / 'numpy_dedup_candidate.py',
    'preblock_resume.py': o / 'preblock_resume.py',
    'run_optimized_pending.py': o / 'run_optimized_pending.py',
    'controller-r04.py': r / 'controller-r04.py',
    'controller_state_io.py': r / 'controller_state_io.py',
    'cpu_exact_gate.py': o / 'cpu_exact_gate.py',
    'gate-acceptance.json': o / 'dedup-acceptance.json',
    'gate-cloud-verification.json': o / 'dedup-tracking/cloud-verification.json',
    'gate-events.jsonl': o / 'dedup-tracking/events.jsonl',
    'gate-provenance.json': o / 'dedup-provenance.json',
    'frozen-scientific-source-r02.tar.gz': r / 'formal-source-r02.tar.gz',
}
for name in ('test_numpy_packing_candidate.py', 'test_numpy_dedup_candidate.py',
             'test_preblock_resume.py', 'numpy_packing_candidate.py', 'git-status.txt', 'git-head.txt'):
    copies[name] = o / name
for name, source in copies.items():
    shutil.copyfile(source, stage / name)

gate = json.loads((stage / 'gate-acceptance.json').read_text())
cloud = json.loads((stage / 'gate-cloud-verification.json').read_text())
events = [json.loads(line) for line in (stage / 'gate-events.jsonl').read_text().splitlines()]
assert events[-1]['result_sha256'] == sha(stage / 'gate-acceptance.json')
assert cloud['verified'] and cloud['state'] == 'FINISHED'
assert gate['candidate_sha256'] == sha(stage / 'numpy_dedup_candidate.py')
old_rows = json.loads((o / 'running-before-optimization.json').read_text())
old_pipelines = [json.loads((o / (t['id'] + '-old-resolved.json')).read_text())['spec']['cpu_pipeline'] for t in old_rows]
assert len(old_rows) == 2 and old_pipelines[0] == old_pipelines[1]
old = old_pipelines[0]
new = dict(old, kind='hf_video_batch_local_frame_dedup_candidate_r02',
           packing_implementation_sha256=gate['candidate_sha256'],
           native_processor_return_tensors='np', tensor_conversion='torch.from_numpy',
           adapter_processor_tensor_type='pt', deduplicate_frames=True,
           dedup_scope='single_video_single_prepare_call', dedup_duplicate_rgb_exact_check=True,
           processed_frame_tail_cache=False, native_processor_unique_group_frames=16)
local_proof = dict(acceptance=binding(stage / 'gate-acceptance.json'),
                   cloud_verification=binding(stage / 'gate-cloud-verification.json'),
                   candidate=binding(stage / 'numpy_dedup_candidate.py'),
                   events=binding(stage / 'gate-events.jsonl'),
                   campaign_sha256=scientific_sha,
                   head_sha256=c['datasets']['videomae']['xd_violence']['head_checkpoint']['sha256'],
                   old_pipeline=old, new_pipeline=new)
validate_execution_proof(local_proof, old, new)
proof = json.loads(json.dumps(local_proof))
for key in ('acceptance', 'cloud_verification', 'candidate', 'events'):
    proof[key]['path'] = remote + '/' + Path(local_proof[key]['path']).name
bindings = {key: dict(proof[proof_key]) for key, proof_key in (
    ('acceptance', 'acceptance'), ('cloud_verification', 'cloud_verification'),
    ('candidate', 'candidate'), ('gate_events', 'events'))}
for key, name in (('resume_implementation', 'preblock_resume.py'), ('launcher', 'run_optimized_pending.py')):
    bindings[key] = {'path': remote + '/' + name, 'sha256': sha(stage / name)}
resume_tasks = {t['id']: {
    'source': c['output_root'] + '/preserved-execution-r02/' + t['id'] + '/features/native',
    'source_resolved_sha256': sha(o / (t['id'] + '-old-resolved.json')),
    'old_swanlab': t['swanlab'], 'old_logger_pid': t['logger_pid'],
    'reason': 'user_authorized_execution_optimization_preserving_complete_videos',
} for t in old_rows}
config = dict(schema='vadbench.preblock-execution-optimization/v1',
              scientific_campaign_sha256=scientific_sha,
              frozen_pipeline_sha256=old['implementation_sha256'],
              original_script=c['snapshot_root'] + '/scripts/icassp2027/run_pre_block_test.py',
              bindings=bindings, resume_execution_proof=proof, resume_tasks=resume_tasks,
              scope='VideoMAE-XD only; independent CPU packing/dedup; no scientific changes',
              formal_efficiency_policy='original native benchmark unchanged; execution throughput separately labeled')
config_path = stage / 'execution-config.json'
config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
archive = o / 'execution-optimization-r03.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for p in sorted(stage.iterdir()):
        tar.add(p, arcname=p.name)
ready = dict(status='ready', scientific_campaign_sha256=scientific_sha,
             candidate_sha256=gate['candidate_sha256'],
             source_bundle=str(archive.resolve()), source_bundle_sha256=sha(archive),
             remote_launcher=remote + '/run_optimized_pending.py',
             remote_config=remote + '/execution-config.json', remote_config_sha256=sha(config_path),
             local_bindings={key: binding(stage / name) for key, name in (
                 ('acceptance', 'gate-acceptance.json'), ('cloud_verification', 'gate-cloud-verification.json'),
                 ('candidate', 'numpy_dedup_candidate.py'), ('events', 'gate-events.jsonl'),
                 ('config', 'execution-config.json'), ('launcher', 'run_optimized_pending.py'),
                 ('resume_implementation', 'preblock_resume.py'), ('controller', 'controller-r04.py'))})
(o / 'execution-ready-r03.json').write_text(json.dumps(ready, indent=2), encoding='utf-8')
print(json.dumps({'bundle_sha256': sha(archive), 'config_sha256': sha(config_path),
                  'candidate_sha256': gate['candidate_sha256'], 'resumable_task_ids': list(resume_tasks)}, indent=2))

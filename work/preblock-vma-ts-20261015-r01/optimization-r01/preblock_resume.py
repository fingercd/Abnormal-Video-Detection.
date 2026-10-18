"""Audited execution-only complete-video resume for this frozen PreBlock campaign.

Not a global FeatureResumeSource replacement. Copy transport only; old sources
remain read-only and old per-forward receipts are retained without alteration.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from vadbench.features import compute_encoder_fingerprint
from vadbench.workflows.feature_resume import FeatureResumeSource
from vadbench.workflows.extraction import _semantic_runtime_identity
from vadbench.token_reduction.pre_block_deployment import PreBlockDeployment


def bound(binding):
    path = Path(binding['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']:
        raise ValueError('resume proof binding SHA mismatch')
    return path


def validate_execution_proof(proof, old_pipeline, new_pipeline):
    """Bind exact tested implementation and successful original cloud gate."""
    receipt = json.loads(bound(proof['acceptance']).read_text())
    cloud = json.loads(bound(proof['cloud_verification']).read_text())
    bound(proof['candidate'])
    events = [json.loads(line) for line in bound(proof['events']).read_text().splitlines() if line.strip()]
    completed = [event for event in events if event.get('kind') == 'completed'
                 and event.get('result_sha256') == proof['acceptance']['sha256']
                 and event.get('metrics', {}).get('qa/completed') == 1]
    if len(completed) != 1 or receipt['tracking_url'] != cloud['url']:
        raise ValueError('resume gate event SHA or original cloud URL differs')
    if (receipt['status'] != 'passed' or not receipt['CPU_only']
            or receipt['encoder_model_loaded'] or receipt['test_truth_accessed']
            or receipt['campaign_sha256'] != proof['campaign_sha256']
            or receipt['head_sha256'] != proof['head_sha256']
            or receipt['candidate_sha256'] != proof['candidate']['sha256']
            or receipt['frozen_pipeline_sha256'] != old_pipeline['implementation_sha256']):
        raise ValueError('resume requires the bound exact CPU execution gate')
    if (not cloud['verified'] or cloud['state'] != 'FINISHED'
            or cloud['completion_summary']['qa/completed']['value'] != 1):
        raise ValueError('resume exactness gate cloud completion missing')
    if old_pipeline != proof['old_pipeline'] or new_pipeline != proof['new_pipeline']:
        raise ValueError('resume pipeline identities differ from proof')
    expected = dict(old_pipeline)
    expected.update(kind='hf_video_ordered_cpu_pipeline_numpy_packing_candidate_r01',
                    packing_implementation_sha256=proof['candidate']['sha256'],
                    native_processor_return_tensors='np', tensor_conversion='torch.from_numpy',
                    adapter_processor_tensor_type='pt')
    if new_pipeline.get('kind') == 'hf_video_batch_local_frame_dedup_candidate_r02':
        expected.update(kind='hf_video_batch_local_frame_dedup_candidate_r02',
                        deduplicate_frames=True, dedup_scope='single_video_single_prepare_call',
                        dedup_duplicate_rgb_exact_check=True, processed_frame_tail_cache=False,
                        native_processor_unique_group_frames=16)
    if new_pipeline != expected:
        raise ValueError('resume permits only the exact NumPy packing override')
    cases = receipt['cases']
    if (len({row['video_id'] for row in cases}) < 4
            or {row['is_anomaly'] for row in cases} != {False, True}
            or {row['requested_batch_size'] for row in cases} != {1, 8}):
        raise ValueError('resume CPU gate coverage incomplete')
    for row in cases:
        if (not row['RGB_exact'] or not row['indices_exact'] or not row['processor_exact']
                or not row['inputs'] or any(not v['torch_equal'] for v in row['inputs'].values())
                or (row['requested_batch_size'] == 8 and not row['two_worker_repeated_exact'])):
            raise ValueError('resume CPU gate contains a nonexact case')


class PreBlockResumeSource(FeatureResumeSource):
    """Reuse original copy/array/source checks with actual PreBlock receipts."""
    def __init__(self, source, destination, resolved, *, encode_context_factory,
                 execution_proof, resume_transport='copy'):
        if resume_transport != 'copy':
            raise ValueError('PreBlock resume supports independent copy only')
        if not isinstance(encode_context_factory, PreBlockDeployment):
            raise ValueError('PreBlock resume requires an actual PreBlockDeployment')
        self.source = Path(source).expanduser().resolve()
        self.destination = Path(destination).resolve()
        if (self.source == self.destination or self.source.is_relative_to(self.destination)
                or self.destination.is_relative_to(self.source)):
            raise ValueError('resume source and destination must not overlap')
        self.transport = 'copy'
        source_bytes = (self.source / 'resolved.json').read_bytes()
        self.source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        self.resolved = json.loads(source_bytes)
        for key in ('data_content_evidence', 'encoder_fingerprint', 'paper_identity'):
            if self.resolved[key] != resolved[key]:
                raise ValueError('resume source ' + key + ' differs')
        old_spec = deepcopy(self.resolved['spec'])
        new_spec = deepcopy(resolved['spec'])
        old_pipeline = old_spec.pop('cpu_pipeline')
        new_pipeline = new_spec.pop('cpu_pipeline')
        # Unlike identity resume, preserve B8 and its original per-batch geometry.
        if old_spec != new_spec:
            raise ValueError('PreBlock resume spec differs beyond CPU execution')
        validate_execution_proof(execution_proof, old_pipeline, new_pipeline)
        self.semantic_runtime = _semantic_runtime_identity(self.resolved['runtime'])
        if self.semantic_runtime != _semantic_runtime_identity(resolved['runtime']):
            raise ValueError('PreBlock resume native runtime differs')
        if compute_encoder_fingerprint({'paper_pooled_cache': self.semantic_runtime}) != resolved['encoder_fingerprint']:
            raise ValueError('PreBlock resume runtime fingerprint does not reproduce')
        self.reducer = old_spec['representation']['reducer']
        if self.reducer != dict(encode_context_factory.reducer_identity):
            raise ValueError('PreBlock resume actual deployment differs from source')
        if (old_spec['runtime_id'] != 'videomae' or self.reducer['encoder_id'] != 'videomae'
                or self.reducer['name'] not in ('attention_topk', 'visionzip_internal', 'uniform')
                or self.reducer['target_block'] not in (6, 9, 12)
                or self.reducer['patch_keep_ratio'] not in (.5, .3)
                or old_spec['micro_batch_size'] != 8):
            raise ValueError('PreBlock resume is restricted to frozen VMA B8 configurations')
        self.depth = len(encode_context_factory.context.blocks)
        self.dim = old_spec['representation']['output_dim']
        self.reduction_template = True  # Base copy_video delegates to our validator.
        self.lineage = self.destination / 'resume_lineage'
        self.lineage.mkdir()
        (self.lineage / 'source_resolved.json').write_bytes(source_bytes)
        if (self.source / 'status.json').is_file():
            (self.lineage / 'source_status.json').write_bytes((self.source / 'status.json').read_bytes())
        (self.lineage / 'execution_equivalence.json').write_text(json.dumps(execution_proof, indent=2))

    def _validate_reduction(self, execution, *, clip_index, clip_count, token_count):
        if not isinstance(execution, dict):
            raise ValueError('PreBlock source has no actual execution receipt')
        r = self.reducer
        target = r['target_block'] - 1
        signal = target - 1
        batch = min(8, clip_count - (clip_index // 8) * 8)
        before, after = r['source_tokens'], r['output_tokens']
        expected = dict(encoder='videomae',
            site='previous_complete_block_output_before_target_LN_attention',
            target_block=r['target_block'], signal_block=r['signal_block'],
            method=r['name'], keep_ratio=r['patch_keep_ratio'],
            before_tokens=before, after_tokens=after,
            inputs={str(i): [batch, before if i < target else after, self.dim] for i in range(self.depth)},
            outputs={str(i): [batch, before if i < signal else after, self.dim] for i in range(self.depth)},
            mlp_inputs={str(i): [batch, before if i < target else after, self.dim] for i in range(self.depth)},
            budget_unit='tubelet_patch', time_frames_preserved=None,
            spatial_packing=None, native_attention_backend_unchanged=True,
            gathered_tokens=after, actual_patch_keep_ratio=after / before,
            source_membership_available=True, mass_used_to_reweight_readout=False)
        geometry = dict(execution)
        forward_count = geometry.pop('forward_count', None)
        if (type(forward_count) is not int or forward_count < 1
                or token_count != after or geometry != expected):
            raise ValueError('PreBlock source receipt violates frozen site/budget/shape')

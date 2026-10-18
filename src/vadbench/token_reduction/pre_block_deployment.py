"""Frozen pre-block identity and persistent per-run intervention lifecycle."""
from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
import torch
from .bridges.pre_block import PreBlockIntervention
from .relation_selection import retained_count


class PreBlockDeployment:
    def __init__(self,bridge,*,target_depth,method,keep_ratio):
        self.context=PreBlockIntervention(bridge,target_depth,method,keep_ratio)
        self.frames=int(bridge._root.config.num_frames)
        spatial=(bridge._root.config.image_size//bridge._root.config.patch_size)**2
        if bridge.receipt().encoder_id=='timesformer':
            before=1+self.frames*spatial;after=1+self.frames*retained_count(spatial,keep_ratio)
        else:
            before=spatial*(self.frames//bridge._root.config.tubelet_size);after=retained_count(before,keep_ratio)
        root=Path(__file__).parent
        self._identity={'name':method,'variant':'previous_complete_block_pre_input_v1','encoder_id':bridge.receipt().encoder_id,
            'site':'previous_complete_block_output_before_target_LN_attention','target_block':target_depth+1,'signal_block':target_depth,
            'patch_keep_ratio':keep_ratio,'source_tokens':before,'output_tokens':after,'readout':'native_unchanged_no_mass_reweighting',
            'budget_unit':'spatial_trajectory' if bridge.receipt().encoder_id=='timesformer' else 'tubelet_patch','budget_rounding':'nearest_half_up',
            'calibration':'none','dominant_contextual_weights':[4,1] if method=='visionzip_internal' else None,
            'implementation_sha256':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),root/'bridges/pre_block.py',root/'relation_selection.py')}}

    @property
    def reducer_identity(self):return deepcopy(self._identity)

    def __call__(self,batch):
        if batch.frames.shape[1]!=self.frames or not bool(batch.valid_mask.all()):raise ValueError('pre-block requires complete native clips')
        return _BatchContext(self)

    def close(self):self.context.close()


class _BatchContext:
    def __init__(self,deployment):self.deployment=deployment
    def __enter__(self):
        if not self.deployment.context.active:self.deployment.context.__enter__()
        return self
    def validate_execution(self):
        result=self.deployment.context.receipt()
        if result['after_tokens']!=self.deployment._identity['output_tokens']:raise ValueError('frozen budget differs')
        return {**result,'gathered_tokens':result['after_tokens'],'actual_patch_keep_ratio':
                (result['after_tokens']-1)/(result['before_tokens']-1) if result['encoder']=='timesformer' else result['after_tokens']/result['before_tokens'],
                'source_membership_available':True,'mass_used_to_reweight_readout':False}
    def __exit__(self,exc_type,exc,tb):
        if exc_type is not None:self.deployment.close()


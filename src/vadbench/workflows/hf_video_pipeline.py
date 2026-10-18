"""Bounded CPU preparation of the unchanged native HF video processor.

Reuses the proven ordered scheduler/continuous decoder, not V2's pixel code.
No frame deduplication, no temporal processed-pixel cache and no CUDA offthread.
"""
from __future__ import annotations
from contextlib import contextmanager
from collections.abc import Mapping
from pathlib import Path
import hashlib
import numpy as np
import torch
from vadbench.integrations.transformers_video import _frame_lists,_invoke_processor,_move_to_device
from .extraction_pipeline import VideoMAEv2CPUPipeline


class HFVideoCPUPipeline(VideoMAEv2CPUPipeline):
    def __init__(self,adapter,*,workers=2,prefetch=2):
        self.adapter=adapter;self.workers=workers;self.prefetch=prefetch
        self.pixel_tail={};self.temporal_cache=False;self.deduplicate=False
        if workers<1 or prefetch<1:raise ValueError('workers/prefetch must be positive')

    @property
    def execution_identity(self):
        return {'kind':'hf_video_ordered_cpu_pipeline_v1','workers':self.workers,'prefetch':self.prefetch,
                'continuous_reader':True,'deduplicate_frames':False,'processed_frame_tail_cache':False,
                'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scheduler_sha256':hashlib.sha256((Path(__file__).parent/'extraction_pipeline.py').read_bytes()).hexdigest()}

    def prepare(self,batch):
        videos,lengths=_frame_lists(batch)
        kwargs=dict(self.adapter.processor_kwargs);kwargs['return_tensors']=self.adapter.processor_tensor_type
        if self.adapter.image_size is not None and 'size' not in kwargs:kwargs['size']={'height':self.adapter.image_size,'width':self.adapter.image_size}
        value=_invoke_processor(self.adapter.processor,videos,kwargs)
        inputs=dict(value) if isinstance(value,Mapping) else {'pixel_values':value.pixel_values if hasattr(value,'pixel_values') else value}
        if self.adapter.processor_tensor_type=='np':inputs={k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in inputs.items()}
        if any(torch.is_tensor(v) and v.is_cuda for v in inputs.values()):raise ValueError('CPU worker emitted a CUDA tensor')
        return inputs,lengths

    @contextmanager
    def prepared_context(self,prepared):
        inputs,lengths=prepared;adapter=self.adapter
        if '_prepare_inputs' in adapter.__dict__:raise ValueError('concurrent HF preparation override')
        adapter._prepare_inputs=lambda batch:(_move_to_device(inputs,adapter.device),lengths)
        try:yield
        finally:del adapter._prepare_inputs

"""Fit-only pipeline acceptance: four CPU streams, then one released GPU.

Runs under an ONLINE SwanLab event-stream launcher. Never reads test scores or
test truth. This is scheduling engineering, not a paper efficiency benchmark.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from transformers import VideoMAEImageProcessor

from vadbench.checkpoints import sha256_file
from vadbench.data.batches import clean_encoder_batch
from vadbench.data.dense_sampling import DenseSamplingPlan
from vadbench.data.manifest import load_manifest_jsonl
from vadbench.data.video import OpenCVVideoReader, _batch_from_reader, build_clip_batch, probe_video
from vadbench.features import atomic_write_json, FeatureStore
from vadbench.integrations.videomaev2_encoder import VideoMAEv2Encoder
from vadbench.paper.evaluation import _make_adapter
from vadbench.token_reduction.bridges import create_observation_bridge
from vadbench.token_reduction.relation_deployment import RelationDeployment
from vadbench.workflows.extraction import (PooledExtractionSpec, extract_pooled_features,
    make_sampling_identity, representation_from_verified_encoder)
from vadbench.workflows.extraction_pipeline import VideoMAEv2CPUPipeline


def emit(kind, **values):
    print('EVENT ' + json.dumps({'kind':kind, **values}, allow_nan=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-qa', required=True, type=Path)
    p.add_argument('--project', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--tracking-url', required=True)
    p.add_argument('--gpu-uuid', required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    old = json.loads(a.previous_qa.read_text())['provenance']
    for key in ('manifest', 'role_lock'):
        if sha256_file(Path(old[key]['path'])) != old[key]['sha256']:
            raise ValueError('fit input SHA differs')
    records = tuple(load_manifest_jsonl(old['manifest']['path']))
    roles = json.loads(Path(old['role_lock']['path']).read_text())['partitions']
    if any(str(r.split.value) != 'train' or roles.get(r.video_id) != 'fit' for r in records):
        raise ValueError('only role-locked fit videos allowed')
    paths = {v['video_id']:Path(v['path']) for v in old['source_videos']}
    for video in old['source_videos']:
        if sha256_file(paths[video['video_id']]) != video['sha256']:
            raise ValueError('fit video SHA differs')
    dataset_root = Path(old['source_videos'][0]['path']).parents[1]
    plan = DenseSamplingPlan(16, 2, 16, 'stride1_if_needed')
    jobs = [(r, tuple(plan.sample(probe_video(paths[r.video_id]).num_frames))) for r in records]
    total = sum(len(s) for _, s in jobs)
    # CPU processor only: no model or CUDA allocation while old task owns GPU2.
    cpu = SimpleNamespace(cfg=SimpleNamespace(num_frames=16),
        processor=VideoMAEImageProcessor.from_pretrained(old['constructor']['model_name'], local_files_only=True))
    cpu._tensor_from_rgb_lists = VideoMAEv2Encoder._tensor_from_rgb_lists.__get__(cpu)
    pixel_max = 0.
    for record, samples in jobs:
        with OpenCVVideoReader(paths[record.video_id]) as baseline, OpenCVVideoReader(paths[record.video_id]) as reader:
            pipeline = VideoMAEv2CPUPipeline(cpu, workers=1, temporal_cache=True)
            with pipeline.batches(reader, record.video_id, samples, 8) as batches:
                for group, batch, context in batches:
                    native = _batch_from_reader(baseline, record.video_id, [s.clip for s in group])
                    np.testing.assert_array_equal(native.frames, batch.frames)
                    np.testing.assert_array_equal(native.frame_indices, batch.frame_indices)
                    expected = cpu._tensor_from_rgb_lists([list(c) for c in native.frames])
                    with context:
                        actual = cpu._tensor_from_rgb_lists(None)
                        pixel_max = max(pixel_max, float((actual - expected).abs().max()))
                        if not torch.equal(actual, expected):
                            raise ValueError('native processor pixels differ')
    emit('pixel_equivalence', metrics={'pipeline/rgb_and_pixels_exact':1,'pipeline/pixel_max_abs':pixel_max})

    def cpu_stream_safe(option):
        workers, temporal_cache = option
        local = SimpleNamespace(cfg=cpu.cfg, processor=cpu.processor)
        local._tensor_from_rgb_lists = VideoMAEv2Encoder._tensor_from_rgb_lists.__get__(local)
        for record, samples in jobs:
            with OpenCVVideoReader(paths[record.video_id]) as reader:
                if workers:
                    pipeline = VideoMAEv2CPUPipeline(local, workers=workers, temporal_cache=temporal_cache)
                    with pipeline.batches(reader, record.video_id, samples, 8) as batches:
                        for group, batch, context in batches:
                            with context: local._tensor_from_rgb_lists(None)
                else:
                    for i in range(0, len(samples), 8):
                        batch = _batch_from_reader(reader, record.video_id, [s.clip for s in samples[i:i+8]])
                        local._tensor_from_rgb_lists([list(c) for c in batch.frames])
        return total

    measurements = {}
    options = {'native':(0,False),'w1':(1,False),'w2':(2,False),'w1cache':(1,True)}
    for name, option in options.items():
        elapsed = []
        for repeat in range(2):
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                counts = list(pool.map(cpu_stream_safe, [option]*4))
            seconds = time.perf_counter() - start
            elapsed.append(seconds)
            emit('cpu_four_streams', variant=name, option=option, repeat=repeat,
                metrics={f'pipeline/cpu4_{name}_clips_per_second':sum(counts)/seconds,
                         f'pipeline/cpu4_{name}_seconds':seconds})
        measurements[name] = {'seconds':elapsed,'median_seconds':statistics.median(elapsed),
            'aggregate_clips_per_second':4*total/statistics.median(elapsed)}
    selected = min(('w1','w2','w1cache'), key=lambda name:measurements[name]['median_seconds'])
    workers, temporal_cache = options[selected]
    config = {'workers':workers,'prefetch':2,'deduplicate':True,'temporal_cache':temporal_cache,
        'scope':'execution_only_fit_engineering_not_method_selection',
        'benchmark_source':str(a.output), 'tracking_url':a.tracking_url}
    atomic_write_json(a.output/'cpu-result.json', {'pixel_exact':True,'measurements':measurements,
        'selected':config,'source_videos':old['source_videos'],'clips_per_stream':total})
    emit('cpu_ready', selected=config)

    start_wait = time.monotonic()
    while True:
        query = subprocess.check_output(['nvidia-smi','-i',a.gpu_uuid,
            '--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
        if not query:
            break
        if any(pid.strip() != '31725' for pid in query.splitlines()):
            raise RuntimeError('GPU2 has unexpected owner; no benchmark launch')
        if time.monotonic()-start_wait > 3*3600:
            raise TimeoutError('GPU2 release not observed within bounded wait')
        time.sleep(10)
    emit('gpu_released', gpu_uuid=a.gpu_uuid)
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    torch.manual_seed(0); np.random.seed(0)
    adapter, definition = _make_adapter(SimpleNamespace(encoder='videomaev2',project=a.project,
        device='cuda:0',processor_tensor_type=None))
    bridge = create_observation_bridge('videomaev2',adapter)
    bridge.model.eval().requires_grad_(False)
    args = dict(sampling_kind='dense',clip_frames=16,frame_stride=2,window_stride=16,short_policy='stride1_if_needed')
    sampling = make_sampling_identity(records,dataset_root=dataset_root,**args)
    first = clean_encoder_batch(build_clip_batch(paths[records[0].video_id], records[0].video_id,[jobs[0][1][0].clip]))
    with torch.inference_mode(), bridge.geometry(first.frame_indices,first.valid_mask) as geometry:
        adapter.encode(first,train=False)
    cases = []
    for method in ('identity','attention_topk','visionzip_internal','uniform'):
        factory = None if method=='identity' else RelationDeployment(bridge,geometry.layout,
            depth=5,method=method,keep_ratio=.5)
        reducer = {'name':'identity','calibration':'none'} if factory is None else factory.reducer_identity
        representation = representation_from_verified_encoder(runtime_id='videomaev2',adapter=adapter,
            verified_encoder_identity=definition['identity'],preprocessing={'profile':getattr(adapter,'preprocess_profile','unknown')},
            readout={'kind':getattr(adapter,'pooling','pooled')},reducer=reducer,output_dim=768,
            precision='float32',position_strategy={'kind':'native'})
        spec = PooledExtractionSpec(runtime_id='videomaev2',verified_encoder_identity=definition['identity'],
            representation=representation,sampling=sampling,micro_batch_size=8,**args)
        native_values = None
        timings = {'native':[],'optimized':[]}
        for repeat in range(2):
            # Reverse order on repeat 2 to reduce cache/order bias.
            for variant in (('native','optimized') if repeat==0 else ('optimized','native')):
                pipeline = None if variant=='native' else VideoMAEv2CPUPipeline(adapter.encoder,workers=workers,temporal_cache=temporal_cache)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                result = extract_pooled_features(spec,adapter=adapter,manifest=records,dataset_root=dataset_root,
                    output_root=a.output/'extractions',run_id=f'{method}-{variant}-r{repeat}',
                    encode_context_factory=factory,cpu_pipeline=pipeline)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-start
                if not result.completed: raise RuntimeError(str(result.failures))
                store = FeatureStore(result.feature_root)
                values = np.stack([store.load_array(r,'pooled') for r in store.records()])
                if native_values is None: native_values=values
                np.testing.assert_array_equal(values,native_values)
                timings[variant].append(seconds)
                emit('gpu_case', method=method,variant=variant,repeat=repeat,
                    metrics={f'pipeline/{method}_{variant}_clips_per_second':total/seconds,
                             f'pipeline/{method}_{variant}_seconds':seconds,
                             'pipeline/pooled_max_abs':float(np.max(np.abs(values-native_values))),
                             'pipeline/peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
        speedup = statistics.median(timings['native'])/statistics.median(timings['optimized'])
        cases.append({'method':method,'timings':timings,'speedup':speedup,'pooled_exact':True})
    cpu_speedup = measurements['native']['median_seconds']/measurements[selected]['median_seconds']
    median_speedup = statistics.median(c['speedup'] for c in cases)
    accepted = cpu_speedup>1.05 and median_speedup>1.05 and min(c['speedup'] for c in cases)>.95
    final = {'status':'accepted' if accepted else 'held_no_proven_speedup','scope':'fit_engineering_only',
        'single_gpu':a.gpu_uuid,'cpu_four_stream_speedup':cpu_speedup,'gpu_median_speedup':median_speedup,
        'cases':cases,'configuration':config,'pixel_exact':True,'tracking_url':a.tracking_url,
        'formal_four_gpu_speedup_verified':False}
    atomic_write_json(a.output/'acceptance.json',final)
    if not accepted: raise RuntimeError('pipeline did not pass throughput acceptance; preserve results')
    atomic_write_json(a.output/'execution-config.json',config)
    emit('completed',acceptance=final,metrics={'qa/completed':1,'pipeline/gpu_median_speedup':median_speedup,
        'pipeline/cpu_four_stream_speedup':cpu_speedup})


if __name__=='__main__':
    main()

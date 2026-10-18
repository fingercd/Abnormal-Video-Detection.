"""CPU-only train-input equivalence gate; launch only under online SwanLab tracker.

Does not load an encoder/model, read test records/truth, or alter production.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def emit(kind, **payload):
    print('EVENT ' + json.dumps({'kind': kind, **payload}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tracking-url', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--candidate-file', type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError('at least two independent preparations per path required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError("CPU-only launch requires CUDA_VISIBLE_DEVICES='' explicitly")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.snapshot / 'src'))
    import numpy as np
    import torch
    from transformers import VideoMAEImageProcessor
    from vadbench.paper.profile import load_project
    from vadbench.paper.urdmu_evaluation import _load_source
    from vadbench.data.dense_sampling import DenseSamplingPlan
    from vadbench.data.video import OpenCVVideoReader, build_clip_batch, _batch_from_reader
    from vadbench.data.batches import clean_encoder_batch
    from vadbench.workflows.extraction_pipeline import ContinuousVideoReader
    from vadbench.workflows.hf_video_pipeline import HFVideoCPUPipeline
    from vadbench.integrations.transformers_video import _frame_lists, _invoke_processor
    torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    campaign = json.loads(args.campaign.read_text(encoding='utf-8'))
    ds = campaign['datasets']['videomae']['xd_violence']
    source = _load_source(SimpleNamespace(trained_run=ds['head_run'], dataset='xd_violence', phase='official_frame'))
    if source.checkpoint_sha256 != ds['head_checkpoint']['sha256']:
        raise ValueError('formal fixed head SHA differs')
    project = load_project(campaign['project'])
    # Mirror evaluation._make_adapter constructor resolution, without creating a model.
    constructor = dict(project.encoder('videomae')['definition']['constructor'])
    if ds['processor_tensor_type'] is not None:
        constructor['processor_tensor_type'] = ds['processor_tensor_type']
    model_path = Path(constructor['model_path'])
    processor = VideoMAEImageProcessor.from_pretrained(
        str(model_path), local_files_only=True, revision=constructor.get('revision'))
    adapter = SimpleNamespace(processor=processor,
                              processor_kwargs=dict(constructor.get('processor_kwargs', {})),
                              processor_tensor_type=constructor.get('processor_tensor_type', 'pt'),
                              image_size=constructor.get('image_size'), device='cpu')
    if adapter.processor_tensor_type != 'pt':
        raise ValueError('this gate targets unchanged omitted-default pt VMA-XD identity')
    original = HFVideoCPUPipeline(adapter, workers=2, prefetch=2)

    def numpy_candidate(ad, batch):
        videos, lengths = _frame_lists(batch)
        kwargs = dict(ad.processor_kwargs)
        kwargs['return_tensors'] = 'np'
        if ad.image_size is not None and 'size' not in kwargs:
            kwargs['size'] = {'height': ad.image_size, 'width': ad.image_size}
        processed = _invoke_processor(ad.processor, videos, kwargs)
        return ({key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                 for key, value in dict(processed).items()}, lengths)

    candidate = numpy_candidate
    if args.candidate_file:
        spec = importlib.util.spec_from_file_location('candidate_cpu_packing', args.candidate_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidate = module.candidate_prepare

    def equal(left, right):
        x, lx = left
        y, ly = right
        if x.keys() != y.keys() or not np.array_equal(lx, ly):
            raise ValueError('processor keys or valid lengths differ')
        receipt = {}
        for key in x:
            a, b = x[key], y[key]
            if not (torch.is_tensor(a) and torch.is_tensor(b) and a.device.type == b.device.type == 'cpu'
                    and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)):
                raise ValueError('non-exact processor output: ' + key)
            receipt[key] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                            'sha256': hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest(),
                            'torch_equal': True}
        return receipt

    plan = DenseSamplingPlan(clip_frames=source.sampling.window['clip_frames'],
                             frame_stride=source.sampling.stride['frame_stride'],
                             window_stride=source.sampling.frame_selection['window_stride'],
                             short_policy=source.sampling.frame_selection['short_video_policy'])
    selected = []
    for label in (False, True):
        records = sorted((r for r in source.fit_records if r.is_anomaly == label
                          and len(plan.sample(r.num_frames)) >= 16), key=lambda r: r.video_id)
        if len(records) < 2:
            raise ValueError('requires two sufficiently long training videos per label')
        selected.extend(records[:2])
    root = source.metadata['source_training_view']['dataset_root']
    rows = []
    emit('started', tracking_url=args.tracking_url, dataset='xd_violence',
         training_videos=len(source.fit_records), selected_video_ids=[r.video_id for r in selected],
         metrics={'optimization/cpu_gate_started': 1})
    for rec in selected:
        path = rec.resolve_path(root)
        samples = plan.sample(rec.num_frames)
        video_sha = sha(path)
        for batch_size in (1, 8):
            n = len(samples)
            groups = [('begin', 0, batch_size), ('middle', max(0, n // 2 - batch_size // 2), batch_size),
                      ('tail_full', n - batch_size, batch_size)]
            if batch_size == 8 and n % 8:
                groups.append(('tail_partial', n - n % 8, n % 8))
            # Same continuous reader across sorted groups exercises forward cursor and tail.
            with ContinuousVideoReader(path) as reader:
                for position, start, count in groups:
                    group = samples[start:start + count]
                    t = time.perf_counter()
                    native = clean_encoder_batch(build_clip_batch(path, rec.video_id, [s.clip for s in group]))
                    native_decode = time.perf_counter() - t
                    t = time.perf_counter()
                    decoded = clean_encoder_batch(_batch_from_reader(reader, rec.video_id, [s.clip for s in group]))
                    continuous_decode = time.perf_counter() - t
                    if not (np.array_equal(native.frames, decoded.frames)
                            and np.array_equal(native.frame_indices, decoded.frame_indices)
                            and np.array_equal(native.valid_lengths, decoded.valid_lengths)):
                        raise ValueError('native and continuous decoder RGB/indices/lengths differ')
                    timings = {'original_prepare_seconds': [], 'optimized_prepare_seconds': []}
                    reference = None
                    inputs_receipt = None
                    for repeat in range(args.repeats):
                        outputs = {}
                        order = ('original', 'optimized') if repeat % 2 == 0 else ('optimized', 'original')
                        for name in order:
                            t = time.perf_counter()
                            outputs[name] = original.prepare(decoded) if name == 'original' else candidate(adapter, decoded)
                            timings[name + '_prepare_seconds'].append(time.perf_counter() - t)
                        inputs_receipt = equal(outputs['original'], outputs['optimized'])
                        if reference is None:
                            reference = outputs['original']
                        else:
                            equal(reference, outputs['original'])
                            equal(reference, outputs['optimized'])
                    row = {'video_id': rec.video_id, 'is_anomaly': rec.is_anomaly,
                           'video_sha256': video_sha, 'requested_batch_size': batch_size,
                           'actual_batch_size': count, 'position': position, 'sample_start': start,
                           'total_video_clips': n, 'frame_indices': decoded.frame_indices.tolist(),
                           'RGB_sha256': hashlib.sha256(decoded.frames.tobytes()).hexdigest(),
                           'RGB_exact': True, 'indices_exact': True, 'processor_exact': True,
                           'inputs': inputs_receipt, 'native_decode_seconds': native_decode,
                           'continuous_decode_seconds': continuous_decode, **timings}
                    row['prepare_speedup'] = statistics.median(timings['original_prepare_seconds']) / statistics.median(timings['optimized_prepare_seconds'])
                    if batch_size == 8:
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            for prepared in pool.map(lambda _: candidate(adapter, decoded), range(4)):
                                equal(reference, prepared)
                        row['two_worker_repeated_exact'] = True
                    rows.append(row)
                    emit('case', result=row, metrics={'optimization/cpu_exact_cases': len(rows),
                         'optimization/prepare_speedup_last_case': row['prepare_speedup']})
    result = {'status': 'passed', 'tracking_url': args.tracking_url, 'CPU_only': True,
              'encoder_model_loaded': False, 'test_truth_accessed': False,
              'role': 'training_input_execution_equivalence_not_quality_or_formal_efficiency',
              'campaign_sha256': sha(args.campaign), 'script_sha256': sha(__file__),
              'candidate_sha256': None if args.candidate_file is None else sha(args.candidate_file),
              'candidate': 'inline_numpy_pack' if args.candidate_file is None else str(args.candidate_file),
              'frozen_pipeline_sha256': sha(args.snapshot / 'src/vadbench/workflows/hf_video_pipeline.py'),
              'head_sha256': source.checkpoint_sha256, 'constructor': constructor,
              'processor_class': type(processor).__module__ + '.' + type(processor).__name__,
              'processor_config_sha256': sha(model_path / 'preprocessor_config.json'),
              'processor_config': processor.to_dict(), 'sampling': source.sampling.to_dict(),
              'repeats_per_path': args.repeats, 'cases': rows,
              'timing_caveat': 'CPU/storage shared; prepare-only excludes decode/GPU; decode timings are descriptive, OS cache/order uncontrolled'}
    (args.output / 'acceptance.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    emit('completed', result_path=str(args.output / 'acceptance.json'),
         result_sha256=sha(args.output / 'acceptance.json'), metrics={'optimization/cpu_gate_completed': 1, 'qa/completed': 1})


if __name__ == '__main__':
    main()

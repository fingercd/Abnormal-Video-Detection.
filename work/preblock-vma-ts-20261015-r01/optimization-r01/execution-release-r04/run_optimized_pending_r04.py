"""Apply an audited execution-only pipeline to a NEW VMA-XD task.

The frozen scientific snapshot and strict head compatibility remain untouched.
This launcher refuses a different task scope and records the actual execution
override separately rather than claiming the old pipeline was executed.
"""
import argparse
import hashlib
import importlib.util
import json
import runpy
import sys
from functools import partial
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution-config', type=Path, required=True)
    parser.add_argument('--execution-config-sha256', required=True)
    parser.add_argument('--script', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if sha(args.execution_config) != args.execution_config_sha256:
        raise ValueError('execution config SHA mismatch')
    config = json.loads(args.execution_config.read_text())
    for binding in config['bindings'].values():
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('execution proof/source binding mismatch')
    receipt = json.loads(Path(config['bindings']['acceptance']['path']).read_text())
    cloud = json.loads(Path(config['bindings']['cloud_verification']['path']).read_text())
    if receipt['status'] != 'passed' or not cloud['verified'] or cloud['state'] != 'FINISHED':
        raise ValueError('exactness and online terminal proof are required')
    if receipt['candidate_sha256'] != config['bindings']['candidate']['sha256']:
        raise ValueError('gate did not test the actual candidate')
    if receipt['campaign_sha256'] != config['scientific_campaign_sha256']:
        raise ValueError('gate tested another scientific campaign')
    if receipt['frozen_pipeline_sha256'] != config['frozen_pipeline_sha256']:
        raise ValueError('gate tested another original CPU pipeline')
    if cloud['completion_summary']['qa/completed']['value'] != 1:
        raise ValueError('online gate completion point missing')
    if receipt['tracking_url'] != cloud['url']:
        raise ValueError('gate result and verified SwanLab run differ')
    events = [json.loads(line) for line in Path(config['bindings']['gate_events']['path']).read_text().splitlines() if line]
    completed = [event for event in events if event['kind'] == 'completed']
    if len(completed) != 1 or completed[0]['result_sha256'] != config['bindings']['acceptance']['sha256']:
        raise ValueError('gate event does not bind the accepted result bytes')
    if str(args.script) != config['original_script']:
        raise ValueError('only frozen original entry point may execute')
    campaign_path = Path(remaining[remaining.index('--campaign') + 1])
    if sha(campaign_path) != config['scientific_campaign_sha256']:
        raise ValueError('scientific campaign changed')
    if remaining[remaining.index('--campaign-sha256') + 1] != config['scientific_campaign_sha256']:
        raise ValueError('command scientific campaign SHA differs')
    campaign = json.loads(campaign_path.read_text())
    task_id = remaining[remaining.index('--task') + 1]
    task = next(t for t in campaign['tasks'] if t['id'] == task_id)
    scope = config['validated_scope']
    if (task['encoder'], task['dataset']) != (scope['encoder'],scope['dataset']):
        raise ValueError('optimization gate belongs to another encoder/dataset')
    if (receipt.get('encoder'),receipt.get('dataset')) != (scope['encoder'],scope['dataset']):
        raise ValueError('actual gate receipt scope differs')
    if receipt['head_sha256'] != campaign['datasets'][task['encoder']][task['dataset']]['head_checkpoint']['sha256']:
        raise ValueError('actual gate used another fixed head')
    if (Path(campaign['output_root']) / task_id).exists():
        raise ValueError('execution override only permitted on never-started tasks')
    candidate = Path(config['bindings']['candidate']['path'])
    spec = importlib.util.spec_from_file_location('audited_numpy_packing', candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from vadbench.workflows import hf_video_pipeline
    expected_pipeline = Path(campaign['snapshot_root']) / 'src/vadbench/workflows/hf_video_pipeline.py'
    if Path(hf_video_pipeline.__file__).resolve() != expected_pipeline.resolve():
        raise ValueError('imported pipeline is not the frozen original module')
    if sha(expected_pipeline) != config['frozen_pipeline_sha256']:
        raise ValueError('original pipeline file changed after the exactness gate')
    # The ordinary class is captured as the candidate's base at module import.
    # No adapter constructor, processor state, model or global HF API is patched.
    hf_video_pipeline.HFVideoCPUPipeline = module.NumpyPackedHFVideoCPUPipeline
    resume = config.get('resume_tasks', {}).get(task_id)
    if resume is not None:
        source = Path(resume['source'])
        if sha(source / 'resolved.json') != resume['source_resolved_sha256']:
            raise ValueError('preserved original feature identity changed')
        resume_path = Path(config['bindings']['resume_implementation']['path'])
        resume_spec = importlib.util.spec_from_file_location('audited_preblock_resume', resume_path)
        resume_module = importlib.util.module_from_spec(resume_spec)
        resume_spec.loader.exec_module(resume_module)
        from vadbench.paper import pre_block_testing
        from vadbench.workflows import feature_resume
        original_extract = pre_block_testing.extract_pooled_features

        def resumed_extract(*positional, **keywords):
            # One extraction call per child. Only the constructor lookup inside
            # that call changes; original array/index/copy checks are inherited.
            if 'resume_source' in keywords:
                raise ValueError('unexpected second resume source')
            keywords.update(resume_source=source, resume_transport='copy')
            original_factory = feature_resume.FeatureResumeSource
            feature_resume.FeatureResumeSource = partial(
                resume_module.PreBlockResumeSource,
                execution_proof=config['resume_execution_proof'])
            try:
                return original_extract(*positional, **keywords)
            finally:
                feature_resume.FeatureResumeSource = original_factory

        pre_block_testing.extract_pooled_features = resumed_extract
    print('EVENT ' + json.dumps({'kind': 'execution_optimization',
          'task_id': task_id, 'execution_config': config,
          'execution_config_sha256': args.execution_config_sha256,
          'metrics': {'execution/numpy_packing_enabled': 1}}), flush=True)
    sys.argv = [str(args.script), *remaining]
    runpy.run_path(str(args.script), run_name='__main__')


if __name__ == '__main__':
    main()

"""Asset organization must preserve frozen identities and failed attempts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_tool(name):
    source = Path(__file__).resolve().parents[2] / 'scripts/icassp2027' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(tmp_path):
    blob = tmp_path / 'features.npz'
    np.savez(blob, features=np.ones((1, 4), dtype=np.float32))
    index = tmp_path / 'index.jsonl'
    index.write_text(json.dumps({'feature_path': blob.name,
        'arrays': {'features': {'sha256': sha(blob)}}}) + '\n', encoding='utf-8')
    contract = tmp_path / 'extraction-contract.json'
    contract.write_text('{"status":"completed"}\n', encoding='utf-8')
    return {'dataset': 'ucf_crime', 'encoder': 'videomaev2', 'method': 'pair_select',
            'feature_contract': {'path': str(contract), 'sha256': sha(contract)},
            'feature_index': {'path': str(index), 'sha256': sha(index)}}


def test_full_index_verification_detects_change_outside_sample_record(tmp_path):
    tool = load_tool('verify_formal_feature_bindings')
    row = binding(tmp_path)
    assert tool.verify(row, full_index_sha=True)['status'] == 'index_verified_sample_blob_verified'
    with (tmp_path / 'index.jsonl').open('a', encoding='utf-8') as target:
        target.write(' \n')
    result = tool.verify(row, full_index_sha=True)
    assert result['status'] == 'failed'
    assert 'index_sha_mismatch' in result['issues']
    assert result['checks']['first_blob_sha_matches'] is True


def test_frozen_contract_mutation_is_rejected(tmp_path):
    tool = load_tool('verify_formal_feature_bindings')
    row = binding(tmp_path)
    (tmp_path / 'extraction-contract.json').write_text('{"status":"failed"}\n', encoding='utf-8')
    assert 'contract_sha_mismatch' in tool.verify(row, full_index_sha=True)['issues']


def test_discovery_does_not_treat_logs_or_stale_completed_run_as_formal(tmp_path):
    tool = load_tool('catalog_existing_assets')
    runs = tmp_path / 'runs'
    campaign = runs / 'dsanet-extension-20260923-r01/test-campaign-r02'
    for name in ('logs', 'state', 'ucf-dense'):
        (campaign / name).mkdir(parents=True)
    kimi = tmp_path / 'kimi'
    stale = kimi / 'compressed-budgetsweep-r01/videomaev2/xd/0p40/merged/pair_select-r01.partial-stale'
    stale.mkdir(parents=True)
    (stale / 'status.json').write_text('{"status":"completed"}', encoding='utf-8')
    assets = tool.collect(argparse.Namespace(kimi_root=str(kimi), runs_root=str(runs),
        dsanet_train_root=str(tmp_path/'dsanet_train'), datasets_root=str(tmp_path/'datasets'),
        framework_root=str(tmp_path/'framework')))
    assert len(assets['features']) == 2
    old = next(row for row in assets['features'] if row['family'] == 'video_vit')
    assert old['summary_status'] == 'completed'
    assert old['role'] == 'historical_attempt'
    assert old['qa_status'] == 'discovered_unverified'

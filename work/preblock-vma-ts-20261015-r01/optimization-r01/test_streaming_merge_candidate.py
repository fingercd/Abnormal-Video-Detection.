"""Synthetic pooled FeatureStore tests, no model or real experiment."""
import hashlib,importlib.util,json,os,shutil,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path.cwd()/'tests/icassp2027'))
from test_feature_resume import _setup,_run,_snapshot
from vadbench.features import FeatureStore
from vadbench.workflows.extraction import _merge_shards
p=Path(__file__).parent/'streaming_merge_candidate.py';spec=importlib.util.spec_from_file_location('sm',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def source(tmp_path):
    records,adapter,spec=_setup(tmp_path,'dense')
    result=_run(tmp_path,records,adapter,spec,'source')
    assert result.completed
    return Path(result.run_dir)


def test_two_stage_merge_exact_bytes_and_fresh_cache(tmp_path):
    root=source(tmp_path);baseline=tmp_path/'baseline';candidate=tmp_path/'candidate'
    shutil.copytree(root,baseline);shutil.copytree(root,candidate)
    before=_snapshot(root);merge=m.StreamingMergeCandidate()
    for video in ('one','two'):
        a=baseline/'shards'/video;b=candidate/'shards'/video
        ablocks=sorted((a/'blocks').iterdir());bblocks=sorted((b/'blocks').iterdir())
        assert _merge_shards(a,ablocks)==merge(b,bblocks)==31
        assert (a/'index.jsonl').read_bytes()==(b/'index.jsonl').read_bytes()
    assert merge.stats['full_bundle_validations']==62
    assert merge.stats['fresh_hash_cache_reuses']==0
    assert _merge_shards(baseline,list((baseline/'shards').iterdir()))==merge(candidate,list(reversed(list((candidate/'shards').iterdir()))))==62
    assert (baseline/'index.jsonl').read_bytes()==(candidate/'index.jsonl').read_bytes()
    assert merge.stats['full_bundle_validations']==62 and merge.stats['fresh_hash_cache_reuses']==62
    assert merge.stats['maximum_open_input_indexes']==1
    assert len(merge.validated)==62 and all(v is True for v in merge.validated.values())
    assert before==_snapshot(root)


@pytest.mark.parametrize('tamper',['unsorted','duplicate','layout','foreign_encoder','foreign_video'])
def test_invalid_source_never_replaces_published_index(tmp_path,tamper):
    root=source(tmp_path);shard=root/'shards/one';index=shard/'index.jsonl';rows=[json.loads(l) for l in index.read_text().splitlines()]
    if tamper=='unsorted':rows[0],rows[1]=rows[1],rows[0]
    elif tamper=='duplicate':rows.insert(1,rows[0])
    elif tamper=='layout':rows[0]['arrays']['features']['shape']=[2,4]
    elif tamper=='foreign_encoder':rows[1]['encoder_fingerprint']='foreign'
    else:rows[1]['video_id']='foreign'
    index.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    target=root/'index.jsonl';old=target.read_bytes()
    with pytest.raises((ValueError,OSError)):m.StreamingMergeCandidate()(root,list((root/'shards').iterdir()))
    assert target.read_bytes()==old
    assert not list(root.glob('.index.jsonl.stream-*.tmp'))


def test_cached_content_tamper_detected_even_same_path_inode_size_mtime(tmp_path):
    root=source(tmp_path);merge=m.StreamingMergeCandidate()
    for shard in (root/'shards').iterdir():merge(shard,list((shard/'blocks').iterdir()))
    shard=root/'shards/one';row=FeatureStore(shard).records()[0]
    blob=FeatureStore(shard)._resolve(row.arrays['features'].path);st=blob.stat();data=bytearray(blob.read_bytes());data[len(data)//2]^=1;blob.write_bytes(data)
    os.utime(blob,ns=(st.st_atime_ns,st.st_mtime_ns))
    assert m._stat(blob)==(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    old=(root/'index.jsonl').read_bytes()
    with pytest.raises(ValueError,match='fresh NPZ checksum'):merge(root,list((root/'shards').iterdir()))
    assert (root/'index.jsonl').read_bytes()==old


def test_no_cache_still_full_bundle_validation(tmp_path):
    root=source(tmp_path);merge=m.StreamingMergeCandidate();merge(root,list((root/'shards').iterdir()))
    assert merge.stats['full_bundle_validations']==62 and merge.stats['fresh_hash_cache_reuses']==0


def test_streaming_structure_does_not_use_full_jsonl_writer():
    code=p.read_text()
    assert 'atomic_write_jsonl' not in code and 'merged.append' not in code
    assert 'stream.write(payload.encode' in code and 'os.fsync' in code and 'os.replace' in code

def test_validation_is_interleaved_with_stream_reads_not_full_materialization(tmp_path,monkeypatch):
    root=source(tmp_path);events=[];original=FeatureStore.iter_records
    def tracked(store):
        for row in original(store):
            events.append(('read',row.clip_id))
            yield row
    monkeypatch.setattr(FeatureStore,'iter_records',tracked)
    merge=m.StreamingMergeCandidate();validate=merge._validate_blob
    def checked(store,row):
        assert events[-1]==('read',row.clip_id)
        events.append(('validate',row.clip_id));validate(store,row)
    merge._validate_blob=checked
    merge(root,list((root/'shards').iterdir()))
    assert sum(kind=='validate' for kind,_ in events)==62

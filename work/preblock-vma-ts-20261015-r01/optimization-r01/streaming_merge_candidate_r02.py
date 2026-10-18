"""R02 single-resolve execution-only streaming merge for this campaign's pooled NPZ shards.

No formal source is patched by importing this file. Instantiate once per child
and install the callable at that child's extraction._merge_shards call site.
Each input index must be internally sorted and contain one encoder/video. All
inputs must share one encoder and occupy noninterleaving sorted key ranges.
"""
from contextlib import closing
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from vadbench.features import FeatureStore


def _key(record):
    return (record.encoder_fingerprint, record.video_id, record.clip_index, record.clip_id)


def _stat(path):
    value=path.stat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)


def _digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


class StreamingMergeCandidate:
    """Run-local validation cache; arrays and full JSON rows are never retained."""
    def __init__(self):
        self.validated={}
        self.stats={'calls':0,'records':0,'full_bundle_validations':0,
                    'fresh_hash_cache_reuses':0,'maximum_open_input_indexes':0}

    def _validate_blob(self,store,record,_validated_path=None):
        if record.storage_format!='npz' or set(record.arrays)!={'features','pooled'}:
            raise ValueError('stream merge is scoped to pooled features/pooled NPZ')
        paths={ref.path for ref in record.arrays.values()}
        digests={ref.sha256 for ref in record.arrays.values()}
        if len(paths)!=1 or len(digests)!=1:
            raise ValueError('stream merge NPZ references disagree on path/digest')
        # Internal callers pass the path already checked in both source and final stores.
        # Direct calls retain a normal store containment resolution.
        path=store._resolve(next(iter(paths))) if _validated_path is None else _validated_path
        digest=next(iter(digests));before=_stat(path)
        layout=tuple((name,ref.key,tuple(ref.shape),ref.dtype,ref.nbytes,ref.sha256)
                     for name,ref in sorted(record.arrays.items()))
        cache_key=(str(path),before,digest,layout)
        if cache_key in self.validated:
            # Fresh content hash is mandatory even when inode/stat are unchanged.
            if _digest(path)!=digest:
                raise ValueError('stream merge fresh NPZ checksum mismatch')
            if _stat(path)!=before:
                raise ValueError('stream merge NPZ changed during checksum')
            self.stats['fresh_hash_cache_reuses']+=1
            return
        arrays=store.load_bundle(record) # Original complete SHA + both ZIP arrays.
        for name,ref in record.arrays.items():
            array=arrays[name]
            if (tuple(array.shape)!=ref.shape or array.dtype.str!=ref.dtype
                    or array.nbytes!=ref.nbytes or array.dtype.kind!='f'
                    or not np.isfinite(array).all()):
                raise ValueError('stream merge actual array layout/dtype/finite differs')
        if (arrays['features'].ndim!=2 or arrays['features'].shape[0]!=1
                or arrays['pooled'].ndim!=1
                or not np.array_equal(arrays['features'][0],arrays['pooled'])):
            raise ValueError('stream merge requires equal pooled-only feature arrays')
        if _stat(path)!=before:
            raise ValueError('stream merge NPZ changed during array validation')
        self.validated[cache_key]=True
        self.stats['full_bundle_validations']+=1

    def __call__(self,run_dir,shard_dirs):
        run_dir=Path(run_dir).resolve()
        descriptors=[]
        encoder=None
        for raw in shard_dirs:
            shard=Path(raw).resolve()
            if shard==run_dir or not shard.is_relative_to(run_dir):
                raise ValueError('stream merge shard escapes destination tree')
            if not (shard/'blobs').is_dir():raise ValueError('stream merge source blob directory absent')
            store=FeatureStore(shard)
            with closing(store.iter_records()) as rows:
                first=next(rows,None)
            if first is None:
                raise ValueError('stream merge requires nonempty published shard indexes')
            if encoder is None:encoder=first.encoder_fingerprint
            if first.encoder_fingerprint!=encoder:
                raise ValueError('stream merge requires one encoder fingerprint')
            descriptors.append((_key(first),shard))
        descriptors.sort(key=lambda item:item[0])
        final=FeatureStore(run_dir)
        fd,temporary=tempfile.mkstemp(prefix='.index.jsonl.stream-',suffix='.tmp',dir=run_dir)
        temporary=Path(temporary)
        previous=None;video=None;seen_ids=set();seen_indices=set();count=0
        try:
            with os.fdopen(fd,'wb') as stream:
                for first_key,shard in descriptors:
                    store=FeatureStore(shard);prefix=shard.relative_to(run_dir).as_posix()
                    source_previous=None
                    self.stats['maximum_open_input_indexes']=max(self.stats['maximum_open_input_indexes'],1)
                    with closing(store.iter_records()) as rows:
                        for record in rows:
                            key=_key(record)
                            if source_previous is None and key!=first_key:
                                raise ValueError('stream merge input changed after descriptor read')
                            if key[:2]!=first_key[:2]:
                                raise ValueError('stream merge input shard is not single encoder/video')
                            if source_previous is not None and key<=source_previous:
                                raise ValueError('stream merge input index is unsorted or duplicate')
                            if previous is not None and key<=previous:
                                raise ValueError('stream merge shard key ranges overlap/duplicate')
                            if video!=key[:2]:
                                video=key[:2];seen_ids.clear();seen_indices.clear()
                            if record.clip_id in seen_ids or record.clip_index in seen_indices:
                                raise ValueError('stream merge duplicate clip identity/index')
                            seen_ids.add(record.clip_id);seen_indices.add(record.clip_index)
                            raw_paths={ref.path for ref in record.arrays.values()}
                            if len(raw_paths)!=1:
                                raise ValueError('stream merge requires literally identical NPZ paths')
                            raw_path=next(iter(raw_paths))
                            source_path=store._resolve(raw_path)
                            rewritten_path=f'{prefix}/{raw_path}'
                            target_path=final._resolve(rewritten_path)
                            if target_path!=source_path:
                                raise ValueError('stream merge rewritten blob path differs')
                            references={name:replace(ref,path=rewritten_path) for name,ref in record.arrays.items()}
                            rewritten=replace(record,arrays=references)
                            self._validate_blob(final,rewritten,_validated_path=target_path)
                            payload=json.dumps(rewritten.to_dict(),ensure_ascii=False,allow_nan=False,
                                               sort_keys=True,separators=(',',':'))
                            stream.write(payload.encode('utf-8')+b'\n')
                            previous=source_previous=key;count+=1
                stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,run_dir/'index.jsonl')
        finally:
            if temporary.exists():temporary.unlink()
        self.stats['calls']+=1;self.stats['records']+=count
        return count

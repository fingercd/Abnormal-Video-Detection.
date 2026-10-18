import os
import pytest
from vadbench.paper import pre_block_efficiency as timing


def test_frozen_condition_grid_has_exact19_unique_conditions():
    rows=timing.condition_grid()
    assert len(rows)==19
    assert rows[0]=={'method':'identity','target_block':None,'keep_ratio':None}
    assert {(r['method'],r['target_block'],r['keep_ratio']) for r in rows[1:]}=={
        (m,b,k) for m in ('attention_topk','visionzip_internal','uniform')
        for b in (6,9,12) for k in (.5,.3)}
    assert len(rows)*2*3*2==228
    assert len(rows)*2*3*2==228


def test_exclusivity_rejects_foreign_pid_and_ignores_other_gpu(monkeypatch):
    monkeypatch.setenv('VAD_GPU_UUID','own-gpu')
    monkeypatch.setenv('VAD_GPU_SLOT','5')
    monkeypatch.setattr(timing.subprocess,'check_output',lambda *a,**kw:
        f'other-gpu, 1827, 12092\nown-gpu, {os.getpid()}, 1000\n')
    result=timing.gpu_receipt()
    assert result['exclusive'] and result['own_pid']==os.getpid()
    monkeypatch.setattr(timing.subprocess,'check_output',lambda *a,**kw:
        f'own-gpu, {os.getpid()+1}, 1000\n')
    with pytest.raises(RuntimeError,match='foreign process'):timing.gpu_receipt()

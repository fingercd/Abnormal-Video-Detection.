import hashlib
import json
from pathlib import Path

import pytest

from vadbench.paper.relation_testing import bound, load_campaign


def campaign(tmp_path):
    marker = tmp_path / "marker.json"
    marker.write_text("{}", encoding="utf-8")
    ref = {"path":str(marker), "sha256":hashlib.sha256(marker.read_bytes()).hexdigest()}
    tasks = [{"id":f"{d}-dense","dataset":d,"method":"identity","block_index":None,"keep_ratio":1.0}
             for d in ("ucf_crime","xd_violence")]
    tasks += [{"id":f"{d}-{m}-{b}-{k}","dataset":d,"method":m,"block_index":b,"keep_ratio":k}
              for d in ("ucf_crime","xd_violence") for m in ("attention_topk","visionzip_internal","uniform")
              for b in (5,8,11) for k in (.5,.3)]
    c={"schema":"vadbench.relation-test-campaign/v1","status":"frozen_before_test", "tasks":tasks,
       "snapshot_root":str(tmp_path),"implementation_sha256":{"marker.json":ref["sha256"]},
       "method_freeze":ref,"freeze_markdown":ref,"engineering_qa":ref}
    return c


def save_load(tmp_path,c):
    p=tmp_path/"campaign.json"
    p.write_text(json.dumps(c),encoding="utf-8")
    return load_campaign(p,hashlib.sha256(p.read_bytes()).hexdigest())


def test_exact_38_condition_campaign(tmp_path):
    assert len(save_load(tmp_path,campaign(tmp_path))["tasks"])==38


@pytest.mark.parametrize("change",["drop","duplicate","budget","depth","method","unfrozen"])
def test_changed_test_grid_fails_closed(tmp_path,change):
    c=campaign(tmp_path)
    if change=="drop":c["tasks"].pop()
    elif change=="duplicate":c["tasks"][-1]=c["tasks"][-2]
    elif change=="budget":c["tasks"][-1]["keep_ratio"]=.4
    elif change=="depth":c["tasks"][-1]["block_index"]=10
    elif change=="method":c["tasks"][-1]["method"]="pair_select"
    else:c["status"]="development"
    with pytest.raises(ValueError):save_load(tmp_path,c)


def test_input_hash_change_rejected(tmp_path):
    p=tmp_path/"file"
    p.write_text("old")
    b={"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    p.write_text("new")
    with pytest.raises(ValueError):bound(b)

import json

import pytest

from scripts.icassp2027.track_relation_qa import paired_quality_gate


@pytest.mark.parametrize("change",[None,"sampling","checkpoint"])
def test_actual_paired_prediction_identities(tmp_path,change):
    a={"checkpoint_sha256":"head-sha","evaluation_sampling_fingerprint":"input-sha"}
    b=dict(a)
    if change=="sampling":b["evaluation_sampling_fingerprint"]="foreign-input"
    if change=="checkpoint":b["checkpoint_sha256"]="foreign-head"
    for name,data in [("ucf_crime-dense",a),("candidate",b)]:
        p=tmp_path/name/"scoring/predictions/resolved.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(data),encoding="utf-8")
    script=paired_quality_gate({"campaign":{"output_root":str(tmp_path),
        "datasets":{"ucf_crime":{"head_checkpoint":{"sha256":"head-sha"}}}},
        "task":{"dataset":"ucf_crime","id":"candidate"}})
    code=script.split("<<'QUALITY_GATE'\n",1)[1].rsplit("QUALITY_GATE\n",1)[0]
    if change:
        with pytest.raises(ValueError):exec(code,{})
    else:
        exec(code,{})

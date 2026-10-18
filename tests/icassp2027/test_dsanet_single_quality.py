"""Synthetic coverage of the one-task identity gate and supplementary metrics."""
import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,auc
from vadbench.workflows import quality_comparison as qc
from vadbench.workflows.dsanet import _quality_single_runtime as module
from vadbench.workflows.dsanet.__main__ import stage_parser


def test_points_use_exact_interval_frame_counts_and_keep_AP_definitions_separate():
    videos=[qc.PairedVideoIntervals('normal',0,4,np.array([[0,4]]),[0],[.5],[.2]),
            qc.PairedVideoIntervals('positive',1,6,np.array([[0,2],[2,6]]),[0,1],[.1,.5],[.4,.8])]
    for index in (0,1):
        points=module.point_metrics(videos,index)
        y=np.array([0]*6+[1]*4)
        s=np.array(([.5]*4+[.1]*2+[.5]*4) if index==0 else ([.2]*4+[.4]*2+[.8]*4))
        p,r,_=precision_recall_curve(y,s)
        assert points['frame_roc_auc']==pytest.approx(roc_auc_score(y,s),abs=1e-15)
        assert points['frame_ap']==pytest.approx(average_precision_score(y,s),abs=1e-15)
        assert points['frame_pr_auc']==pytest.approx(auc(r,p),abs=1e-15)
        assert points['CI'].startswith('not_computed')


def test_bootstrap_seed_is_explicit_without_changing_legacy_default():
    labels=np.array([0,0,1,1])
    np.testing.assert_array_equal(qc._stratified_weights(labels),qc._stratified_weights(labels,qc.BOOTSTRAP_SEED))
    np.testing.assert_array_equal(qc._stratified_weights(labels,0),qc._stratified_weights(labels,0))
    assert not np.array_equal(qc._stratified_weights(labels,0),qc._stratified_weights(labels))


def test_single_quality_identity_failure_precedes_any_truth_read(tmp_path):
    root=tmp_path/'runs';(root/'ucf-dense/scores').mkdir(parents=True)
    (root/'ucf-dense/prediction-completed.json').write_text(json.dumps({'status':'completed'}))
    (root/'ucf-dense/scores/summary.json').write_text(json.dumps({'status':'completed','videos':290,'checkpoint_sha256':'wrong','manifest_sha256':'m'}))
    c={'output_root':str(root),'datasets':{'ucf':{'videos':290,'checkpoint':{'sha256':'expected'},'manifest':{'sha256':'m'},'truth':{'path':'DO_NOT_OPEN'}}}}
    with pytest.raises(ValueError,match='head or manifest'):
        module.paired_identity(c,{'task_id':'ucf-uniform-b6-remove50','dataset':'ucf'})


def test_new_parser_is_one_task_not_legacy_tier_loop():
    o=stage_parser('quality-one').parse_args(['--campaign','c.json','--task','t','--output','preview','--dense-cache','cache'])
    assert o.task=='t' and o.output==Path('preview') and not hasattr(o,'tiers')

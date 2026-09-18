"""Only synthetic labels/scores; brute frame expansion is a small-test oracle."""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from sklearn.metrics import auc, precision_recall_curve

from vadbench.metrics import average_precision_score, roc_auc_score
from vadbench.paper import quality_comparison as qc
from vadbench.research.xd_feature_grid import precision_recall_trapezoid_auc

SHA = {"synthetic_scores_and_labels": "0" * 64}


def video(name, weak, labels, dense, method):
    n = len(labels)
    return qc.PairedVideoIntervals(name, weak, n, np.column_stack([np.arange(n), np.arange(1, n + 1)]), labels, dense, method)


def cohort():
    return [
        video("n1", 0, [0], [.4], [.2]),
        video("n2", 0, [0] * 7, [.2, .2, .8, .8, -.1, .6, .2], [.1, .5, .5, .5, .2, .8, .2]),
        video("p1", 1, [1, 1], [.4, .4], [.9, .5]),
        video("p2", 1, [0, 1, 1, 0, 0, 1, 0, 1], [.9, .4, .6, .4, .2, .6, .4, .9], [.6, .6, .9, .2, .3, .8, .8, .1]),
    ]


def expand(videos, weights, method):
    labels, scores = [], []
    for v, copies in zip(videos, weights, strict=True):
        lengths = np.diff(np.asarray(v.intervals), axis=1).ravel()
        y = np.repeat(v.labels, lengths)
        s = np.repeat(getattr(v, method), lengths)
        labels.extend([y] * int(copies))
        scores.extend([s] * int(copies))
    return np.concatenate(labels), np.concatenate(scores)


def reference(videos, weights, method, metric):
    y, s = expand(videos, weights, method)
    if not y.any() or y.all():
        return np.nan  # comparison policy requires BOTH frame classes
    if metric == "frame_roc_auc":
        return roc_auc_score(y, s)
    value = precision_recall_trapezoid_auc(y, s)
    precision, recall, _ = precision_recall_curve(y, s)
    assert value == pytest.approx(auc(recall, precision), abs=2e-15)
    return value


def histograms(videos, method_index):
    output = []
    for v in videos:
        lengths, labels, scores = qc._validated_video(v)
        output.append(qc._histogram(scores[method_index], lengths, labels))
    return output


@pytest.mark.parametrize("metric", ["frame_roc_auc", "frame_pr_auc"])
def test_weighted_engines_equal_brute_frame_repetition_with_ties(metric):
    videos = cohort()
    draws = np.random.default_rng(42).multinomial(4, [.25] * 4, size=300)
    # Include duplicated mixed-label video, missing highest score thresholds,
    # a fully positive-only draw, and a fully negative-only draw explicitly.
    draws[:5] = [[0, 0, 0, 4], [2, 0, 1, 1], [0, 0, 4, 0], [4, 0, 0, 0], [1, 1, 1, 1]]
    dense, method = histograms(videos, 0), histograms(videos, 1)
    if metric == "frame_roc_auc":
        p, n = np.array([h.positive.sum() for h in dense]), np.array([h.negative.sum() for h in dense])
        actual = qc._auc_delta(draws, qc._auc_pair_matrix(method) - qc._auc_pair_matrix(dense), p, n)
    else:
        actual = qc._PRCurve.prepare(method).evaluate(draws) - qc._PRCurve.prepare(dense).evaluate(draws)
    expected = np.array([reference(videos, w, "method_scores", metric) - reference(videos, w, "dense_scores", metric) for w in draws])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-15, equal_nan=True)


@pytest.mark.parametrize("metric", ["frame_roc_auc", "frame_pr_auc"])
def test_public_point_draws_counts_and_order_invariance(metric):
    videos = cohort()
    result = qc.compare_paired_quality(videos, metric=metric, source_sha256=SHA)
    point_dense = reference(videos, np.ones(4, dtype=int), "dense_scores", metric)
    point_method = reference(videos, np.ones(4, dtype=int), "method_scores", metric)
    assert result["dense_value"] == pytest.approx(point_dense, abs=2e-15)
    assert result["method_value"] == pytest.approx(point_method, abs=2e-15)
    assert result["delta_method_minus_dense"] == pytest.approx(point_method - point_dense, abs=2e-15)
    assert result["n_videos"] == 4 and result["n_frames"] == 18
    assert result["weak_label_strata"] == {"normal": 2, "positive": 2}
    assert result["bootstrap_frame_count_range"] == [6, 30]
    assert result["valid_draws"] == result["bootstrap_draws"] == 10000
    assert result["invalid_draws"] == 0
    assert result["source_sha256"] == SHA
    reordered = qc.compare_paired_quality(list(reversed(videos)), metric=metric, source_sha256=SHA)
    assert result == reordered


def test_stratified_draws_keep_group_sizes_and_are_shared_by_methods():
    labels = np.array([0, 0, 1, 1, 1])
    weights = qc._stratified_weights(labels)
    assert weights.shape == (10000, 5)
    assert np.all(weights[:, :2].sum(1) == 2)
    assert np.all(weights[:, 2:].sum(1) == 3)
    assert (weights > 1).any() and (weights == 0).any()
    np.testing.assert_array_equal(weights, qc._stratified_weights(labels))
    results = [qc.compare_paired_quality(cohort(), metric=m, source_sha256=SHA) for m in ("frame_roc_auc", "frame_pr_auc")]
    assert results[0]["bootstrap_weights_sha256"] == results[1]["bootstrap_weights_sha256"]


@pytest.mark.parametrize("metric", ["frame_roc_auc", "frame_pr_auc"])
def test_public_ci_matches_brute_force_for_every_possible_stratified_composition(metric):
    videos = cohort()
    result = qc.compare_paired_quality(videos, metric=metric, source_sha256=SHA)
    # Two videos in each stratum give only nine distinct compositions. Compute
    # every one by physically concatenating its frames, then independently
    # reconstruct the predeclared 10000-draw distribution from that lookup.
    exact = {}
    for n1 in range(3):
        for p1 in range(3):
            w = (n1, 2 - n1, p1, 2 - p1)
            exact[w] = reference(videos, w, "method_scores", metric) - reference(videos, w, "dense_scores", metric)
    rng = np.random.default_rng(20260918)
    w = np.column_stack([rng.multinomial(2, [.5, .5], size=10000), rng.multinomial(2, [.5, .5], size=10000)])
    draws = np.array([exact[tuple(row)] for row in w])
    np.testing.assert_allclose([result["ci_low"], result["ci_high"]], np.quantile(draws, [.025, .975]), atol=3e-15, rtol=0)


def test_pr_is_trapezoid_not_step_ap_and_handles_tied_thresholds():
    videos = [video("n", 0, [0], [.5], [.1]), video("p", 1, [1], [.5], [.9])]
    result = qc.compare_paired_quality(videos, metric="frame_pr_auc", source_sha256=SHA)
    assert result["dense_value"] == .75
    assert result["method_value"] == 1.
    assert average_precision_score([0, 1], [.5, .5]) == .5
    assert result["delta_method_minus_dense"] == .25
    assert result["noninferiority"] == "supported"


@pytest.mark.parametrize("metric", ["frame_roc_auc", "frame_pr_auc"])
def test_identical_methods_have_zero_paired_ci_not_independent_method_draws(metric):
    videos = [replace(v, method_scores=v.dense_scores) for v in cohort()]
    result = qc.compare_paired_quality(videos, metric=metric, source_sha256=SHA)
    assert result["delta_method_minus_dense"] == result["ci_low"] == result["ci_high"] == 0.
    assert result["noninferiority"] == "supported"


@pytest.mark.parametrize("metric", ["frame_roc_auc", "frame_pr_auc"])
def test_invalid_positive_class_draws_are_not_redrawn_or_hidden(metric):
    videos = [video("n", 0, [0], [.2], [.2]), video("p_empty", 1, [0], [.2], [.2]), video("p_event", 1, [1], [.8], [.8])]
    result = qc.compare_paired_quality(videos, metric=metric, source_sha256=SHA)
    assert result["status"] == "NA" and result["noninferiority"] == "NA"
    assert result["reason"] == "fewer_than_99.9_percent_valid_draws"
    assert result["point_noninferiority_pass"] is True
    assert result["ci_low"] is result["ci_high"] is None
    assert result["valid_draws"] + result["invalid_draws"] == 10000
    assert result["invalid_reasons"]["no_positive_frames"] == result["invalid_draws"] > 0


@pytest.mark.parametrize("metric", ["frame_roc_auc", "frame_pr_auc"])
def test_no_negative_draws_and_absent_weak_normal_stratum_are_explicit(metric):
    videos = [video("p_all", 1, [1, 1], [.6, .7], [.7, .6]), video("p_mixed", 1, [0, 1], [.1, .8], [.2, .9])]
    result = qc.compare_paired_quality(videos, metric=metric, source_sha256=SHA)
    assert result["weak_label_strata"] == {"normal": 0, "positive": 2}
    assert result["status"] == "NA"
    assert result["invalid_reasons"]["no_negative_frames"] == result["invalid_draws"] > 0


@pytest.mark.parametrize("metric", ["frame_roc_auc", "frame_pr_auc"])
@pytest.mark.parametrize("label", [0, 1])
def test_one_frame_class_is_policy_na_even_if_standalone_pr_defines_value(metric, label):
    videos = [video("v", label, [label, label], [.1, .2], [.2, .1])]
    result = qc.compare_paired_quality(videos, metric=metric, source_sha256=SHA)
    assert result["status"] == "NA"
    assert result["reason"] == "both_frame_classes_required_by_comparison_protocol"
    assert result["dense_value"] is result["method_value"] is None
    assert result["valid_draws"] == 0 and result["invalid_draws"] == 10000


def test_exact_noninferiority_boundary_and_point_only_inconclusive():
    equal = np.full(10000, -.005)
    assert qc._interval_and_decision(equal, -.005)["noninferiority"] == "supported"
    one_ulp = np.full(10000, np.nextafter(-.005, -np.inf))
    one_ulp_result = qc._interval_and_decision(one_ulp, float(one_ulp[0]))
    assert one_ulp_result["noninferiority"] == "supported"
    assert one_ulp_result["ci_low"] < -.005
    assert one_ulp_result["decision_ci_low"] == -.005
    worse = np.full(10000, -.005 - 2 * qc.DECISION_TOLERANCE)
    assert qc._interval_and_decision(worse, float(worse[0]))["noninferiority"] == "inferior"
    crossing = np.linspace(-.02, .02, 10000)
    result = qc._interval_and_decision(crossing, 0.)
    assert result["point_noninferiority_pass"] is True
    assert result["noninferiority"] == "inconclusive"


@pytest.mark.parametrize("negative_frames,expected_decision", [(199, "inferior"), (200, "supported"), (201, "supported")])
def test_public_roc_boundary_point_and_ci_use_the_same_paired_delta(negative_frames, expected_decision):
    videos = [
        qc.PairedVideoIntervals("normal", 0, negative_frames, [[0, 1], [1, negative_frames]], [0, 0], [.1, .1], [.9, .1]),
        qc.PairedVideoIntervals("positive", 1, 1, [[0, 1]], [1], [.8], [.5]),
    ]
    result = qc.compare_paired_quality(videos, metric="frame_roc_auc", source_sha256=SHA)
    assert result["dense_value"] == 1.
    assert result["delta_method_minus_dense"] == result["ci_low"] == result["ci_high"] == -1 / negative_frames
    assert result["point_noninferiority_pass"] is (expected_decision == "supported")
    assert result["noninferiority"] == expected_decision
    if negative_frames == 200:
        assert result["method_value"] == .995
        assert result["delta_method_minus_dense"] == -.005


@pytest.mark.parametrize("frames", [np.int64(2 ** 62), np.uint64(2 ** 63)])
def test_numpy_integer_frame_count_cannot_overflow_exact_count_guard(frames):
    intervals = np.array([[0, frames]], dtype=frames.dtype)
    videos = [
        qc.PairedVideoIntervals("normal", 0, frames, intervals, [0], [.1], [.1]),
        qc.PairedVideoIntervals("positive", 1, frames, intervals, [1], [.9], [.9]),
    ]
    with pytest.raises(ValueError, match="exact float64 integer range"):
        qc.compare_paired_quality(videos, metric="frame_roc_auc", source_sha256=SHA)


@pytest.mark.parametrize("case", ["near_one", "midrange"])
def test_public_pr_exact_boundary_preserves_raw_values_and_discloses_decision_precision(case):
    if case == "near_one":
        videos = [
            qc.PairedVideoIntervals("normal", 0, 1, [[0, 1]], [0], [.1], [.5]),
            qc.PairedVideoIntervals("positive", 1, 99, [[0, 99]], [1], [.9], [.5]),
        ]
        exact_dense, exact_method = Fraction(1), Fraction(199, 200)
    else:
        videos = [
            qc.PairedVideoIntervals("normal", 0, 24, [[0, 19], [19, 24]], [0, 0], [.9, .1], [.5, .5]),
            qc.PairedVideoIntervals("positive", 1, 1, [[0, 1]], [1], [.9], [.5]),
        ]
        exact_dense, exact_method = Fraction(21, 40), Fraction(13, 25)
    assert exact_method - exact_dense == Fraction(-1, 200)
    result = qc.compare_paired_quality(videos, metric="frame_pr_auc", source_sha256=SHA)
    assert result["dense_value"] == float(exact_dense)
    assert result["method_value"] == float(exact_method)
    assert result["delta_method_minus_dense"] == result["ci_low"] == result["ci_high"] == float(exact_method) - float(exact_dense)
    assert result["ci_low"] < -.005  # unmodified arithmetic output is visible
    assert result["decision_point_delta"] == result["decision_ci_low"] == result["decision_ci_high"] == -.005
    assert result["point_noninferiority_pass"] is True and result["noninferiority"] == "supported"
    assert result["noninferiority_margin"] == .005
    assert result["decision_precision"]["ulps"] == 8
    assert result["decision_precision"]["absolute_tolerance"] == 8 * np.spacing(1.)


@pytest.mark.parametrize("metric", ["frame_roc_auc", "frame_pr_auc"])
@pytest.mark.parametrize("offset,expected", [(-1, "supported"), (0, "supported"), (1, "inferior")])
def test_public_real_degradation_just_outside_numeric_resolution_is_not_tolerated(metric, offset, expected):
    total = 200000000
    if metric == "frame_roc_auc":
        failed_pairs = 1000000 + offset
        videos = [
            qc.PairedVideoIntervals("normal", 0, total, [[0, failed_pairs], [failed_pairs, total]], [0, 0], [.1, .1], [.9, .1]),
            qc.PairedVideoIntervals("positive", 1, 1, [[0, 1]], [1], [.8], [.5]),
        ]
        exact_delta = Fraction(-failed_pairs, total)
    else:
        negatives = 2000000 + offset
        positives = total - negatives
        videos = [
            qc.PairedVideoIntervals("normal", 0, negatives, [[0, negatives]], [0], [.1], [.5]),
            qc.PairedVideoIntervals("positive", 1, positives, [[0, positives]], [1], [.9], [.5]),
        ]
        exact_delta = Fraction(-negatives, 2 * total)
    result = qc.compare_paired_quality(videos, metric=metric, source_sha256=SHA)
    assert result["noninferiority"] == expected
    assert result["point_noninferiority_pass"] is (expected == "supported")
    assert result["delta_method_minus_dense"] == pytest.approx(float(exact_delta), abs=2e-16, rel=0)
    if offset:
        assert abs(float(exact_delta) + .005) > 1000000 * qc.DECISION_TOLERANCE
        assert result["decision_ci_low"] == result["ci_low"]


def test_failed_draw_percentile_bounds_retain_all_draws_and_9990_gate():
    values = np.linspace(-.01, .02, 10000)
    values[:10] = np.nan
    result = qc._interval_and_decision(values, .005)
    assert result["status"] == "available"
    assert result["ci_low"] == np.quantile(np.nan_to_num(values, nan=-1.), .025)
    assert result["ci_high"] == np.quantile(np.nan_to_num(values, nan=1.), .975)
    assert result["ci_low"] < np.quantile(values[np.isfinite(values)], .025)
    assert result["ci_high"] > np.quantile(values[np.isfinite(values)], .975)
    values[10] = np.nan
    result = qc._interval_and_decision(values, .005)
    assert result["status"] == "NA" and result["noninferiority"] == "NA"


@pytest.mark.parametrize("metric", ["frame_roc_auc", "frame_pr_auc"])
def test_rare_invalid_draws_meet_threshold_without_silent_filtering(metric):
    videos = [video("n", 0, [0], [.1], [.1]), video("p_empty", 1, [0], [.2], [.2])]
    videos += [video(f"p{i}", 1, [1], [.8], [.8]) for i in range(4)]
    result = qc.compare_paired_quality(videos, metric=metric, source_sha256=SHA)
    assert 0 < result["invalid_draws"] <= 10
    assert result["valid_draws"] >= 9990 and result["status"] == "available"
    assert result["ci_low"] == result["ci_high"] == 0.
    assert "endpoint bounds" in result["ci_rule"]


@pytest.mark.parametrize("metric", ["frame_roc_auc", "frame_pr_auc"])
def test_billion_frame_intervals_are_compressed_not_expanded(metric, monkeypatch):
    videos = [
        qc.PairedVideoIntervals("n", 0, 10**9, [[0, 10**9]], [0], [.1], [.1]),
        qc.PairedVideoIntervals("p", 1, 10**9, [[0, 500000000], [500000000, 10**9]], [0, 1], [.2, .9], [.2, .9]),
    ]
    monkeypatch.setattr(np, "repeat", lambda *a, **k: pytest.fail("frame expansion is forbidden"))
    monkeypatch.setattr(np, "tile", lambda *a, **k: pytest.fail("frame expansion is forbidden"))
    result = qc.compare_paired_quality(videos, metric=metric, source_sha256=SHA)
    assert result["n_frames"] == 2 * 10**9 and result["n_videos"] == 2
    assert result["compressed_video_score_rows"] == {"dense": 3, "method": 3}
    assert result["dense_value"] == result["method_value"] == 1.


def test_bootstrap_pr_engine_never_sorts_again(monkeypatch):
    curve = qc._PRCurve.prepare(histograms(cohort(), 0))
    weights = qc._stratified_weights(np.array([0, 0, 1, 1]))
    monkeypatch.setattr(np, "argsort", lambda *a, **k: pytest.fail("per-draw sorting is forbidden"))
    monkeypatch.setattr(np, "unique", lambda *a, **k: pytest.fail("per-draw sorting is forbidden"))
    assert np.isfinite(curve.evaluate(weights)).all()


@pytest.mark.parametrize("mutation", ["duplicate", "gap", "overlap", "tail", "score_length", "nan", "label", "weak_normal", "bool_weak"])
def test_malformed_or_unpaired_inputs_are_rejected(mutation):
    videos = cohort()
    target = videos[-1]
    if mutation == "duplicate":
        videos.append(videos[0])
    elif mutation in ("gap", "overlap", "tail"):
        intervals = target.intervals.copy()
        if mutation == "gap":
            intervals[0, 1] = 2
            intervals[1, 0], intervals[1, 1] = 3, 4
        elif mutation == "overlap":
            intervals[1, 0] = 0
        else:
            intervals[-1, 1] += 1
        videos[-1] = replace(target, intervals=intervals)
    elif mutation == "score_length":
        videos[-1] = replace(target, method_scores=[.2])
    elif mutation == "nan":
        videos[-1] = replace(target, method_scores=[np.nan] * target.num_frames)
    elif mutation == "label":
        videos[-1] = replace(target, labels=[2] * target.num_frames)
    elif mutation == "weak_normal":
        videos[-1] = replace(target, weak_label=0)
    else:
        videos[-1] = replace(target, weak_label=True)
    with pytest.raises(ValueError):
        qc.compare_paired_quality(videos, metric="frame_roc_auc", source_sha256=SHA)


def test_source_binding_and_metric_name_cannot_silently_select_step_ap():
    with pytest.raises(ValueError, match="step AP"):
        qc.compare_paired_quality(cohort(), metric="frame_ap", source_sha256=SHA)
    with pytest.raises(ValueError, match="SHA-256"):
        qc.compare_paired_quality(cohort(), metric="frame_pr_auc", source_sha256={})

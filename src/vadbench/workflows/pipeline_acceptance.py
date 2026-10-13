"""Execution acceptance; VisionZip is bounded by independent native repeats."""
from __future__ import annotations
import math


def case_equivalent(case):
    if case['method'] != 'visionzip_internal':
        return case['pooled_exact']
    if case['pooled_exact']:
        return True
    noise = case['native_repeat_max_abs']
    cross = case['native_optimized_max_abs']
    relative = case['relative_l2_max']
    return (case.get('numerical_policy') == 'native_repeatability_bound_v1'
            and all(math.isfinite(v) for v in (noise, cross, relative))
            and 0 < noise <= 1e-5
            and cross <= min(1e-5, 2 * noise + 1e-7)
            and relative <= 1e-6)


def validate_acceptance(acceptance):
    cases = acceptance['cases']
    if (acceptance['status'] != 'accepted' or not acceptance['pixel_exact']
            or not math.isfinite(acceptance['cpu_four_stream_speedup'])
            or not math.isfinite(acceptance['gpu_median_speedup'])
            or acceptance['cpu_four_stream_speedup'] <= 1.05
            or acceptance['gpu_median_speedup'] <= 1.05
            or len(cases) != 4
            or {c['method'] for c in cases} != {'identity','attention_topk','visionzip_internal','uniform'}
            or not all(case_equivalent(c) and math.isfinite(c['speedup']) and c['speedup'] > .95 for c in cases)):
        raise ValueError('fit pixel/output/throughput acceptance not passed')

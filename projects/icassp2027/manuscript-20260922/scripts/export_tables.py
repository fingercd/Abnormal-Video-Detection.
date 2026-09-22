#!/usr/bin/env python3
"""Regenerate paper tables from supplied quality exports and explicit pending data.
No model inference is performed. Empty cells remain visibly pending.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENCODERS = ('videomaev2', 'videomae', 'timesformer')
NAMES = {'videomaev2': 'VideoMAEv2-B', 'videomae': 'VideoMAE-B', 'timesformer': 'TimeSformer-B'}
DATASETS = ('ucf_crime', 'xd_violence')
DS = {'ucf_crime': 'UCF-Crime', 'xd_violence': 'XD-Violence'}

def quotas(n: int, r: float) -> list[int]:
    full, tail = divmod(n,10)
    return [int(math.floor(10*r+.5))]*full + ([int(math.floor(tail*r+.5))] if tail else [])

def kept(enc: str, r: float) -> int:
    if enc == 'timesformer':
        return 8*(14*(sum(quotas(196,r))//14))+1
    return sum(quotas(1568,r))

def joint(n: int, d: int = 768) -> int:
    return 12*n*d*d + 2*n*n*d

def divided(p: int, t: int = 8, d: int = 768) -> int:
    return (17*p*t+4*t+8)*d*d + 2*d*(p*t*t+t*(p+1)**2)

def blocks_gflops(enc: str, r: float) -> tuple[float,float]:
    n = 1569 if enc == 'timesformer' else 1568
    k = n if r == 1 else kept(enc,r)
    dense = divided(196) if enc=='timesformer' else joint(n)
    reduced = divided((k-1)//8) if enc=='timesformer' else joint(k)
    return 2*12*dense/1e9, 2*(6*dense+6*reduced)/1e9

def fmt(v: float | None, digits: int = 2) -> str:
    return r'\pending' if v is None else f'{v:.{digits}f}'

def sign(v: float, digits: int = 2) -> str:
    # Preserve the sign of the small but nonzero TimeSformer/UCF difference.
    if abs(v) < 0.005: return f'{v:+.3f}'
    return f'{v:+.{digits}f}'

def main() -> None:
    with (ROOT/'data/quality_matrix_detail.csv').open(newline='') as f:
        source = list(csv.DictReader(f))
    rows = {(r['dataset'],r['encoder'],r['method']): r for r in source}
    for ds in DATASETS:
        for enc in ENCODERS:
            r=rows[ds,enc,'pair_select']
            for field in ('dense_value','method_value','ci_low','ci_high','delta_pp'):
                assert math.isfinite(float(r[field])), (ds,enc,field)
            assert int(r['dense_clips'])==int(r['compressed_clips'])
            assert abs((float(r['method_value'])-float(r['dense_value']))*100-float(r['delta_pp']))<1e-9
    extended=json.loads((ROOT/'data/budget_results.json').read_text())
    profiles=json.loads((ROOT/'data/efficiency_results.json').read_text())
    # Reject accidental unit conversion, unsupported values, or unattributed measurements.
    for ds in DATASETS:
        for enc in ENCODERS:
            base=100*float(rows[ds,enc,'pair_select']['dense_value'])
            for ratio,cell in extended[ds][enc].items():
                value=cell['quality']
                if value is None:
                    continue
                if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=100:
                    raise ValueError(f'{ds}/{enc}/{ratio}: quality must be finite on the 0--100 scale')
                if not cell.get('source'):
                    raise ValueError(f'{ds}/{enc}/{ratio}: provide the source result path')
                if cell.get('delta_pp') is not None and abs(value-base-cell['delta_pp'])>1e-6:
                    raise ValueError(f'{ds}/{enc}/{ratio}: delta_pp does not match quality minus dense')
                lo,hi=cell.get('ci_low_pp'),cell.get('ci_high_pp')
                if (lo is None)!=(hi is None) or (lo is not None and (not math.isfinite(lo) or not math.isfinite(hi) or lo>hi)):
                    raise ValueError(f'{ds}/{enc}/{ratio}: invalid paired interval')
    numeric=('full_encoder_gflops','latency_ms_per_clip','throughput_clips_per_s','peak_allocated_gib','peak_reserved_gib')
    for enc,budgets in profiles.items():
        for ratio,cell in budgets.items():
            present=[key for key in numeric if cell.get(key) is not None]
            for key in present:
                value=cell[key]
                if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                    raise ValueError(f'{enc}/{ratio}/{key}: positive finite measurement required')
            if present and not cell.get('source'):
                raise ValueError(f'{enc}/{ratio}: measurement source is required')
    outputs={}
    q=[]
    for ds in DATASETS:
        q.append(r'\multicolumn{7}{@{}l}{\textit{'+DS[ds]+(' --- frame ROC-AUC' if ds=='ucf_crime' else ' --- frame trapezoidal PR-AUC')+r'}} \\')
        for enc in ENCODERS:
            r=rows[ds,enc,'pair_select']
            c=extended[ds][enc]
            q.append(f"{NAMES[enc]} & {float(r['dense_value'])*100:.2f} & {fmt(c['0.80']['quality'])} & {float(r['method_value'])*100:.2f} & {fmt(c['0.40']['quality'])} & ${sign(float(r['delta_pp']))}$ & $[{float(r['ci_low'])*100:+.2f},\\,{float(r['ci_high'])*100:+.2f}]$ "+r'\\')
        if ds != DATASETS[-1]: q.append(r'\midrule')
    outputs['quality_rows.tex']='\n'.join(q)
    a=[]
    for ds in DATASETS:
        a.append(r'\multicolumn{4}{@{}l}{\textit{'+DS[ds]+r'}} \\')
        for enc in ENCODERS:
            vals=[sign(float(rows[ds,enc,m]['delta_pp'])) for m in ('pair_select','group_uniform','group_random')]
            a.append(f"{NAMES[enc]} & ${vals[0]}$ & ${vals[1]}$ & ${vals[2]}$ "+r'\\')
        if ds != DATASETS[-1]: a.append(r'\midrule')
    outputs['reference_rows.tex']='\n'.join(a)
    ap=[]
    for enc in ENCODERS:
        r=rows['xd_violence',enc,'pair_select']
        ap.append(f"{NAMES[enc]} & {float(r['xd_average_precision_dense'])*100:.2f} & {float(r['xd_average_precision_method'])*100:.2f} & ${sign((float(r['xd_average_precision_method'])-float(r['xd_average_precision_dense']))*100)}$ "+r'\\')
    outputs['ap_rows.tex']='\n'.join(ap)
    theory=[]; all_theory=[]
    for enc in ('videomae','timesformer'):
        theory.append(r'\multicolumn{5}{@{}l}{\textit{'+('VideoMAE / VideoMAEv2' if enc=='videomae' else 'TimeSformer (including CLS)')+r'}} \\')
        for r in (.8,.6,.4):
            n=1569 if enc=='timesformer' else 1568;k=kept(enc,r);d,c=blocks_gflops(enc,r)
            theory.append(f"{round((1-r)*100)}\\% & {k} & {100*(1-k/n):.2f}\\% & {n/k:.3f}$\\times$ & {100*(1-c/d):.2f}\\% "+r'\\')
            all_theory.append(dict(encoder=enc,keep_ratio=r,native_tokens=n,retained_tokens=k,actual_removal_percent=100*(1-k/n),token_compression_factor=n/k,dense_principal_block_gflops=d,reduced_principal_block_gflops=c,principal_block_reduction_percent=100*(1-c/d),mac_to_flops=2,scope='principal_12_transformer_blocks_only_excludes_selector'))
        if enc=='videomae':theory.append(r'\midrule')
    outputs['theory_rows.tex']='\n'.join(theory)
    eff=[]
    for enc in ENCODERS:
        for r in (1.,.8,.6,.4):
            p=profiles[enc][f'{r:.2f}']; n=1569 if enc=='timesformer' else 1568
            d,c=blocks_gflops(enc,r); tok=n if r==1 else kept(enc,r)
            budget='Dense' if r==1 else f'{round((1-r)*100)}\\%'
            eff.append(f"{NAMES[enc] if r==1 else ''} & {budget} & {tok} & {c:.2f} & {fmt(p['full_encoder_gflops'])} & {fmt(p['latency_ms_per_clip'])} & {fmt(p['throughput_clips_per_s'])} & {fmt(p['peak_allocated_gib'])}/{fmt(p['peak_reserved_gib'])} "+r'\\')
        if enc!=ENCODERS[-1]:eff.append(r'\midrule')
    outputs['efficiency_rows.tex']='\n'.join(eff)
    for name,text in outputs.items():
        (ROOT/'tables'/name).write_text('% GENERATED by scripts/export_tables.py; do not edit numerical cells.\n'+text+'\n')
    (ROOT/'data/analytical_costs.json').write_text(json.dumps(all_theory,indent=2)+'\n')
    with (ROOT/'data/quality_ladder_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dataset','encoder','keep_ratio','actual_tokens','quality_0_100','delta_pp','ci_low_pp','ci_high_pp','source'])
        writer.writeheader()
        for ds in DATASETS:
            for enc in ENCODERS:
                original=rows[ds,enc,'pair_select']
                for ratio in (.8,.6,.4):
                    if ratio==.6:
                        cell=dict(quality=100*float(original['method_value']),delta_pp=float(original['delta_pp']),ci_low_pp=100*float(original['ci_low']),ci_high_pp=100*float(original['ci_high']),source='quality_matrix_detail.csv')
                    else:
                        cell=extended[ds][enc][f'{ratio:.2f}']
                    writer.writerow(dict(dataset=ds,encoder=enc,keep_ratio=ratio,actual_tokens=kept(enc,ratio),quality_0_100=cell.get('quality'),delta_pp=cell.get('delta_pp'),ci_low_pp=cell.get('ci_low_pp'),ci_high_pp=cell.get('ci_high_pp'),source=cell.get('source')))
    print('Generated',len(outputs),'tables and the full ladder CSV. Missing measurements remain explicit placeholders.')

if __name__=='__main__':
    main()

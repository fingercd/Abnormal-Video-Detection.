#!/usr/bin/env python3
"""Rebuild the four-system paper from immutable r02 JSON/CSV snapshots.
No model inference or new performance measurement is performed.
Each exported numerical field is recorded with its source field and SHA-256.
"""
from __future__ import annotations
import csv, hashlib, json, math, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SNAP=ROOT/'evidence_snapshot'
E=('videomaev2','videomae','timesformer','clip')
SYSTEMS={'videomaev2':'VideoMAEv2--UR-DMU','videomae':'VideoMAE--UR-DMU','timesformer':'TimeSformer--UR-DMU','clip':'CLIP--DSANet'}
DS=('ucf_crime','xd_violence')
R=(1.,.8,.6,.4)
trace=[];hashes={}
def sha(rel):
 if rel not in hashes: hashes[rel]=hashlib.sha256((SNAP/rel).read_bytes()).hexdigest()
 return hashes[rel]
def load(rel):return json.loads((SNAP/rel).read_text())
def record(claim, rel, field, value, operation='identity'):
 trace.append(dict(claim=claim,evidence_file='evidence_snapshot/'+rel,field=field,value=value,operation=operation,sha256=sha(rel)))
 return value
def take(o,rel,keys,claim,scale=1):
 v=o
 for k in keys:v=v[k]
 return record(claim,rel,'.'.join(str(k) for k in keys),float(v)*scale,f'multiply by {scale}')
def quotas(n,r):
 f,t=divmod(n,10)
 return f*int(10*r+.5)+(int(t*r+.5) if t else 0)
def tokens(enc,r):
 n=197 if enc=='clip' else (1569 if enc=='timesformer' else 1568)
 if r==1:return n
 if enc=='timesformer':return 8*(14*(quotas(196,r)//14))+1
 return quotas(196,r)+1 if enc=='clip' else quotas(1568,r)
def joint(n,d=768):return 12*n*d*d+2*n*n*d
def divided(p,t=8,d=768):return (17*p*t+4*t+8)*d*d+2*d*(p*t*t+t*(p+1)**2)
def flops(enc,r):
 n=tokens(enc,1);k=tokens(enc,r)
 full=divided(196) if enc=='timesformer' else joint(n)
 reduced=divided((k-1)//8) if enc=='timesformer' else joint(k)
 return 2*(6*full+6*reduced)/1e9

def main():
 rows={}
 rel='evidence/three-video-encoders/references/quality_matrix_detail.csv'
 with (SNAP/rel).open(newline='') as f:
  original=[x for x in csv.DictReader(f) if x['method']=='pair_select']
 for x in original:
  ds,enc=x['dataset'],x['encoder'];label=f'{ds}/{enc}/keep0.6'
  fieldprefix=f'row[dataset={ds},encoder={enc},method=pair_select].'
  def c(field,scale=1):return record(label+'/'+field,rel,fieldprefix+field,float(x[field])*scale,f'multiply by {scale}')
  vals=dict(primary=c('method_value',100),ap=c('xd_average_precision_method',100),delta=c('delta_pp'),ci_low=c('ci_low',100),ci_high=c('ci_high',100))
  rows[ds,enc,.6]=vals
  rows[ds,enc,1.]=dict(primary=c('dense_value',100),ap=c('xd_average_precision_dense',100),delta=0,ci_low=None,ci_high=None)
  assert int(x['dense_clips'])==int(x['compressed_clips'])
  for r in (.8,.4):
   raw=f'evidence/three-video-encoders/quality/{ds}/{enc}/0p{int(r*100):02d}_pair_select.json';o=load(raw);label=f'{ds}/{enc}/keep{r}'
   q=take(o,raw,['metric','method_value'],label+'/primary',100)
   dense=take(o,raw,['metric','dense_value'],label+'/dense',100)
   assert abs(dense-rows[ds,enc,1.]['primary'])<1e-8
   ap=take(o,raw,['frame_ap_step','method'],label+'/step_ap',100)
   assert abs(float(o['frame_ap_step']['dense'])*100-rows[ds,enc,1.]['ap'])<1e-8
   rows[ds,enc,r]=dict(primary=q,ap=ap,delta=q-dense,ci_low=take(o,raw,['metric','ci_low'],label+'/ci_low',100),ci_high=take(o,raw,['metric','ci_high'],label+'/ci_high',100))
 for ds,short in zip(DS,('ucf','xd')):
  rel=f'evidence/dsanet/results_published/quality-{short}.json';o=load(rel)
  assert o['checkpoint_origin']=='published' and o['primary_branch']=='coarse'
  pm='frame_roc_auc' if short=='ucf' else 'frame_ap'
  for r in (.8,.6,.4):
   keys=['results',f'coarse-{r}'];label=f'{ds}/clip/keep{r}'
   q=take(o,rel,keys+[pm,'method_value'],label+'/primary',100)
   dense=take(o,rel,keys+[pm,'dense_value'],label+'/dense',100)
   ap=take(o,rel,keys+['frame_ap','method_value'],label+'/step_ap',100)
   dap=take(o,rel,keys+['frame_ap','dense_value'],label+'/dense_step_ap',100)
   rows[ds,'clip',r]=dict(primary=q,ap=ap,delta=q-dense,ci_low=take(o,rel,keys+[pm,'ci_low'],label+'/ci_low',100),ci_high=take(o,rel,keys+[pm,'ci_high'],label+'/ci_high',100))
   if short=='xd':
    rows[ds,'clip',r]['trap']=take(o,rel,keys+['frame_pr_auc','method_value'],label+'/trap_pr_auc',100)
   if (ds,'clip',1.) in rows:assert abs(rows[ds,'clip',1.]['primary']-dense)<1e-8
   rows[ds,'clip',1.]=dict(primary=dense,ap=dap,delta=0,ci_low=None,ci_high=None)
   if short=='xd':rows[ds,'clip',1.]['trap']=take(o,rel,keys+['frame_pr_auc','dense_value'],label+'/dense_trap_pr_auc',100)
 # A primary/secondary pair makes the two XD integration conventions explicit.
 qtex=[]
 for ds in DS:
  panel='UCF-Crime: ROC-AUC / step AP (primary: ROC-AUC)' if ds=='ucf_crime' else 'XD-Violence: trapezoidal PR-AUC / step AP (primary: bold)'
  qtex.append(r'\multicolumn{6}{@{}l}{\textit{'+panel+r'}}\\')
  for enc in E:
   cells=[]
   for r in R:
    v=rows[ds,enc,r]
    if ds=='xd_violence' and enc=='clip':cell=rf"{v['trap']:.2f} / \textbf{{{v['ap']:.2f}}}"
    else:cell=rf"\textbf{{{v['primary']:.2f}}} / {v['ap']:.2f}"
    cells.append(cell)
   v=rows[ds,enc,.6]
   qtex.append(SYSTEMS[enc]+' & '+' & '.join(cells)+rf" & [{v['ci_low']:+.2f}, {v['ci_high']:+.2f}] \\")
  if ds==DS[0]:qtex.append(r'\midrule')
 (ROOT/'tables/quality_rows.tex').write_text('% GENERATED; do not edit numerical cells.\n'+'\n'.join(qtex)+'\n')
 theory=[]
 for enc,name in [('videomaev2','VideoMAEv2-B + UR-DMU'),('videomae','VideoMAE-B + UR-DMU'),('timesformer','TimeSformer-B + UR-DMU'),('clip','CLIP ViT-B/16 + DSANet')]:
  theory.append(name+' & '+' & '.join(str(tokens(enc,r)) for r in R)+r' \\')
 (ROOT/'tables/theory_rows.tex').write_text('% GENERATED\n'+'\n'.join(theory)+'\n')
 # Encoder measurements use separate B1 and B32 experiments, never fabricated B32 memory.
 efficiency=[]; erel='evidence/three-video-encoders/efficiency-summary.json';small=load(erel)
 for enc in E[:3]:
  rel=f'evidence/batch-scaling/results/{enc}/result.json';o=load(rel)
  for r in R:
   key='dense' if r==1 else f'keep_{r:.2f}';ks=['summary','32',key];lab=f'{enc}/keep{r}'
   latency=take(o,rel,ks+['median_wall_ms_per_batch'],lab+'/b32_ms')
   throughput=take(o,rel,ks+['throughput_clips_per_s'],lab+'/b32_throughput')
   speed=take(o,rel,ks+['speedup_vs_dense'],lab+'/b32_speedup')
   sk=['encoders',enc,'budgets',f'{r:.2f}','batch_sizes','1']
   m=dict(enc=enc,cohort='',keep=r,tokens=tokens(enc,r),block_gflops=flops(enc,r),b32_ms=latency,b32_throughput=throughput,b32_speedup=speed,
    b1_ms=take(small,erel,sk+['latency_ms_per_batch'],lab+'/b1_ms'),
    b1_speedup=take(small,erel,sk+['speedup_vs_dense'],lab+'/b1_speedup'),
    b1_alloc_gib=take(small,erel,sk+['peak_allocated_gib'],lab+'/b1_alloc_gib'),
    b1_reserved_gib=take(small,erel,sk+['peak_reserved_gib'],lab+'/b1_reserved_gib'))
   assert abs(32000/latency-throughput)<1e-7
   efficiency.append(m)
 for short in ('ucf','xd'):
  rel=f'evidence/dsanet/benchmark/{short}-summary.json';o=load(rel)
  for r in R:
   tag='None' if r==1 else str(r);lab=f'clip/{short}/keep{r}'
   key=['results',f'batch32-keep{tag}'];sk=['results',f'batch1-keep{tag}']
   m=dict(enc='clip',cohort=short,keep=r,tokens=tokens('clip',r),block_gflops=take(o,rel,key+['principal_block_gflops_per_image'],lab+'/block_gflops'),
    b32_ms=take(o,rel,key+['gpu_p50_ms_median_of_processes'],lab+'/b32_ms'),b32_throughput=take(o,rel,key+['throughput_images_per_second_median'],lab+'/b32_throughput'),b32_speedup=take(o,rel,key+['speedup_vs_dense_median'],lab+'/b32_speedup'),
    b1_ms=take(o,rel,sk+['gpu_p50_ms_median_of_processes'],lab+'/b1_ms'),b1_speedup=take(o,rel,sk+['speedup_vs_dense_median'],lab+'/b1_speedup'),b1_alloc_gib=take(o,rel,sk+['peak_allocated_mib_median'],lab+'/b1_alloc_gib',1/1024),b1_reserved_gib=take(o,rel,sk+['peak_reserved_mib_median'],lab+'/b1_reserved_gib',1/1024))
   assert abs(m['block_gflops']-flops('clip',r))<1e-7
   efficiency.append(m)
 # Full-video speedups: median of matched-repeat dense/reduced ratios.
 pipeline=[]
 for short in ('ucf','xd'):
  times={};memory={}
  for r in R:
   tag='dense' if r==1 else str(r);vals=[];mem=[]
   for rep in range(3):
    rel=f'evidence/dsanet/results_published/pipeline-exclusive-r02/{short}-{tag}-r{rep}.json';o=load(rel)
    assert o['status']=='completed' and o['videos']==16 and 'to_cpu_scores' in o['scope']
    vals.append(take(o,rel,['total_wall_seconds'],f'{short}/keep{r}/rep{rep}/pipeline_wall_s'))
    mem.append(take(o,rel,['peak_gpu_allocated_bytes'],f'{short}/keep{r}/rep{rep}/pipeline_alloc_gib',1/(1024**3)))
   times[r]=vals;memory[r]=max(mem)
  for r in R:
   pipeline.append(dict(dataset=short,keep=r,wall_s=statistics.median(times[r]),speedup=statistics.median([a/b for a,b in zip(times[1.],times[r])]),peak_alloc_gib=memory[r]))
 def marked(value, group, field, decimals, lower=False):
  displayed=round(value[field],decimals)
  best=(min if lower else max)(round(item[field],decimals) for item in group)
  formatted=f"{value[field]:.{decimals}f}"
  return r'\textbf{'+formatted+'}' if displayed==best else formatted
 # Display one CLIP cohort, using XD training inputs like the video-encoder B32 runs.
 displayed_efficiency=[v for v in efficiency if v['enc']!='clip' or v['cohort']=='xd']
 et=[]
 for start in range(0,len(displayed_efficiency),4):
  group=displayed_efficiency[start:start+4]
  if et:et.append(r'\midrule')
  for i,v in enumerate(group):
   label=SYSTEMS[v['enc']] if i==0 else ''
   delete='Dense' if v['keep']==1 else f"{round(100*(1-v['keep']))}\\%"
   values=[
    marked(v,group,'block_gflops',2,lower=True),
    marked(v,group,'b32_ms',2,lower=True),
    marked(v,group,'b32_throughput',2),
    marked(v,group,'b32_speedup',3),
   ]
   et.append(label+' & '+delete+' & '+' & '.join(values)+r' \\')
 (ROOT/'tables/efficiency_rows.tex').write_text('% GENERATED; bold compares budgets within each system.\n'+'\n'.join(et)+'\n')
 (ROOT/'data/four_systems_results.json').write_text(json.dumps(dict(quality=[dict(dataset=ds,encoder=e,keep=r,**v) for (ds,e,r),v in rows.items()],efficiency=efficiency,pipeline=pipeline),indent=2)+'\n')
 with (ROOT/'data/quality_ladder_summary.csv').open('w',newline='') as f:
  fields=['dataset','encoder','keep_ratio','primary_metric','quality_0_100','step_ap_0_100','delta_pp','ci_low_pp','ci_high_pp'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for ds in DS:
   for enc in E:
    for r in R:
     v=rows[ds,enc,r];metric='ROC-AUC' if ds=='ucf_crime' else ('step AP' if enc=='clip' else 'trapezoidal PR-AUC')
     w.writerow(dict(dataset=ds,encoder=enc,keep_ratio=r,primary_metric=metric,quality_0_100=v['primary'],step_ap_0_100=v['ap'],delta_pp=v['delta'],ci_low_pp=v['ci_low'],ci_high_pp=v['ci_high']))
 with (ROOT/'data/numeric_provenance.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
 with (ROOT/'data/evidence_sha256.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['evidence_file','sha256'])
  for p in sorted(SNAP.rglob('*')):
   if p.is_file():w.writerow([str(p.relative_to(ROOT)),hashlib.sha256(p.read_bytes()).hexdigest()])
 print(f'Generated four-system tables: {len(rows)} quality rows, {len(efficiency)} encoder rows, {len(pipeline)} pipeline rows, {len(trace)} numerical provenance entries.')
if __name__=='__main__':main()

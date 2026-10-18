"""Strict final52/48 assembler. Reads existing results only; never reruns ML, bootstrap or timing."""
import argparse,csv,hashlib,json,subprocess,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CAMPAIGN_SHA='16cec7a9cba5ca1505d36a1bb8be6815948bbce72864fe93b5285db7002b584e'
SCOPE_SHA='ba513144248751835cab0b1bfb65b1a1c1ed2a7870e654abc25d192d4115df0d'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readiness(q,quality,scope):
 active=set(scope['active_formal_task_ids']);wanted=set(scope['active_quality_task_ids']);excluded=set(scope['excluded_task_ids'])
 assert (len(active),len(wanted),len(excluded))==(52,48,24)
 assert len(q['tasks'])==76 and len(quality['jobs'])==72
 assert {t['id'] for t in q['tasks']}==active|excluded
 assert all(t['status']=='cancelled_user_scope_change' for t in q['tasks'] if t['id'] in excluded)
 missing_formal=[t['id'] for t in q['tasks'] if t['id'] in active and t['status']!='prediction_completed']
 missing_quality=[r['task']['id'] for r in quality['jobs'] if r['task']['id'] in wanted and r['status']!='completed']
 assert {r['task']['id'] for r in quality['jobs'] if r['task']['id'] in wanted}==wanted
 assert all(r['status']=='cancelled_user_scope_change' for r in quality['jobs'] if r['task']['id'] in excluded)
 return {'ready':not missing_formal and not missing_quality,'formal_completed':52-len(missing_formal),'quality_completed':48-len(missing_quality),'missing_formal':missing_formal,'missing_quality':missing_quality,'cancelled_uniform':24}

def historical_contract_location(contract):
 original='/users/fotile/VAD-icassp2027/outputs/icassp2027/control/official-fulltrain-20260918-r01/ucf-view/contract.json'
 copy='/users/fotile/icassp2027-runs/project/outputs/icassp2027/control/official-fulltrain-20260918-r01/ucf-view/contract.json'
 expected='47467e6c88497fbb8f81bc7e2be59cc9e7ed0b6076cdf1548f0e1fe1be5c6998'
 if contract['path']==original:
  assert contract['sha256']==expected
  return copy,{'original_declared_path':original,'original_sha256_unchanged':expected,'verified_existing_copy':copy,'original_path_missing':True,'identity_rule':'same original bytes SHA; no contract regenerated or modified'}
 return contract['path'],None

def audit_native_receipt(d,encoder):
 assert d['status']=='passed' and d['encoder']==encoder and not d['formal_test_truth_accessed'] and not d['head_training']
 assert len(d['checks'])==36 and set(d['training_inputs'])=={'ucf_crime','xd_violence'}
 assert all(len(v)==4 for v in d['training_inputs'].values())
 seen=set()
 for x in d['checks']:
  target=x['target_block'];keep=x['keep_ratio'];before=1569 if encoder=='timesformer' else 1568;after=({.5:785,.3:473} if encoder=='timesformer' else {.5:784,.3:470})[keep]
  assert target in (6,9,12) and x['signal_block']==target-1 and x['site']=='previous_complete_block_output_before_target_LN_attention'
  assert x['before_tokens']==before and x['after_tokens']==after and x['forward_count']==4
  for block in range(12):
   expected=before if block<target-1 else after
   assert x['inputs'][str(block)][1]==expected and x['mlp_inputs'][str(block)][1]==expected
  if encoder=='timesformer':assert x['budget_unit']=='spatial_trajectory' and x['time_frames_preserved']==8 and x['native_attention_backend_unchanged'] and x['spatial_packing']=='logical_Sx1_not_physical_image_grid'
  seen.add((x['dataset'],x['method'],target,keep))
 assert len(seen)==36
 return {'encoder':encoder,'conditions':36,'retained_method_conditions':24,'training_inputs_per_dataset':4,'target_blocks':[6,9,12],'signal_blocks':[5,8,11],'actual_prefix_and_target_MLP_input_shapes_checked':True,'formal_test_truth_accessed':False,'new_head_training':False}

def verify_remote_bindings(bindings,aliases,runs,cancelled_outputs):
 # Only exact paths already bound by the original completed quality comparisons.
 code="""import hashlib,json,os
from pathlib import Path
assert os.uname().nodename=='ibnode3'
expected=EXPECTED
aliases=ALIASES
for actual,declared in aliases.items():
 if Path(actual).resolve()!=Path(declared).resolve():raise ValueError('declared canonical alias changed:'+actual)
for name,digest in expected.items():
 p=Path(name);h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 if h.hexdigest()!=digest:raise ValueError('original source artifact changed:'+name)
runs=RUNS
for item in runs:
 d=Path(item['remote_run'])
 if (d/'exit-code.txt').read_text().strip()!='0':raise ValueError('original remote run not exit0:'+item['id'])
 if (d/'swanlab-url.txt').read_text().strip()!=item['swanlab_url']:raise ValueError('original remote cloud identity differs:'+item['id'])
 pid=int((d/'pid.txt').read_text());proc=Path('/proc')/str(pid)
 if proc.exists() and str(d/'run.sh') in (proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode():raise ValueError('original task PID still live:'+item['id'])
for root in CANCELLED_OUTPUTS:
 d=Path(root)
 if (d/'extraction-contract.json').exists() or (d/'scoring/predictions/predictions.jsonl').exists():raise ValueError('cancelled uniform has unexpected formal output:'+root)
print(json.dumps({'host':os.uname().nodename,'verified_files':len(expected),'all_original_bindings_match':True,'active_remote_exit0_and_same_original_cloud':len(runs),'cancelled_uniform_formal_outputs_absent':len(CANCELLED_OUTPUTS)}))
""".replace('EXPECTED',repr(bindings)).replace('ALIASES',repr(aliases)).replace('RUNS',repr(runs)).replace('CANCELLED_OUTPUTS',repr(cancelled_outputs))
 result=subprocess.run(['ssh','-T','-p','12345','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','fotile@121.196.228.153','python3','-'],input=code.encode(),capture_output=True,timeout=300)
 if result.returncode:raise RuntimeError('final original-file SHA audit failed; do not replay work: '+result.stderr.decode(errors='replace'))
 receipt=json.loads(result.stdout);assert receipt['all_original_bindings_match'] and receipt['verified_files']==len(bindings)
 return receipt

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');parser.add_argument('--resume-report',action='store_true');parser.add_argument('--output',type=Path,default=ROOT/'reports/final-active52');args=parser.parse_args()
 assert sha(ROOT/'campaign-r02.json')==CAMPAIGN_SHA and sha(ROOT/'user-scope-20261016-no-uniform.json')==SCOPE_SHA
 c=read(ROOT/'campaign-r02.json');scope=read(ROOT/'user-scope-20261016-no-uniform.json');q=read(ROOT/'queue.json');quality=read(ROOT/'quality-control-r01/queue.json');ready=readiness(q,quality,scope)
 if args.check_only:print(json.dumps(ready));return
 if not ready['ready']:raise RuntimeError('full user-revised scope incomplete: '+json.dumps(ready))
 original_report_link=None
 if args.output.exists():
  if not args.resume_report:raise FileExistsError('existing final artifact; explicit original-report resume required')
  original_report_link=read(args.output/'swanlab.json')
  assert original_report_link['run_id']=='s2esfubw' and original_report_link['path']=='ryfrcd693/VAD-PreBlock-VMA-TS-20261015/s2esfubw'
  assert not (args.output/'final-results.json').exists(),'do not overwrite an already assembled final report'
 elif args.resume_report:raise ValueError('original report identity missing')
 index=read(ROOT/'quality-results-index.json');index=[x for x in index if x['task']['id'] in scope['active_quality_task_ids']]
 assert len(index)==48 and {x['task']['id'] for x in index}==set(scope['active_quality_task_ids'])
 points=read(ROOT/'quality-cloud-points-verification.json');point_by={p['run_id']:p for p in points};bindings={};aliases={};results={};historical_contract_relocations={}
 for item in index:
  assert sha(item['path'])==item['sha256'];d=read(item['path']);m=d['metric'];task=item['task'];assert d['campaign_task']==task
  assert m['bootstrap_seed']==0 and m['bootstrap_draws']==10000 and m['noninferiority_margin']==.005
  assert m['metric']==('frame_roc_auc' if task['dataset']=='ucf_crime' else 'frame_pr_auc')
  proof=point_by[item['swanlab']['run_id']];assert proof['point_metrics_match'] and proof['local_result_sha256']==item['sha256']
  assert d['source_sha256']==m['source_sha256']
  ds=c['datasets'][task['encoder']][task['dataset']]
  for name,digest in d['source_sha256'].items():
   if name in bindings:assert bindings[name]==digest
   if not name.startswith((c['output_root']+'/',c['control_root']+'/')):
    declared=next((entry['path'] for entry in (ds['truth'],ds['head_checkpoint']) if Path(name).name==Path(entry['path']).name and digest==entry['sha256']),None)
    if declared is None:raise ValueError('unexpected external source binding:'+name)
    aliases[name]=declared
   bindings[name]=digest
  results[task['id']]=d
 for encoder_datasets in c['datasets'].values():
  for dataset in encoder_datasets.values():
   for artifact in (dataset['head_checkpoint'],dataset['truth']):
    if artifact['path'] in bindings:assert bindings[artifact['path']]==artifact['sha256']
    bindings[artifact['path']]=artifact['sha256']
 eff_path=ROOT/'reports/efficiency-active-scope-20261016/efficiency-active.json';eff=read(eff_path);assert eff['scope_override_sha256']==SCOPE_SHA and eff['scientific_campaign_sha256']==CAMPAIGN_SHA
 assert all(eff['modes'][k]['active_groups']==156 for k in ['encoder','pipeline'])
 for mode in eff['modes'].values():assert sha(mode['audited_source_path'])==mode['audited_source_sha256']
 assert sha(ROOT/'efficiency-original-cloud-final-verification.json')==eff['cloud_proof_sha256']
 # Original456 proof remains historical; final tables include only active156+156.
 import swanlab
 api=swanlab.Api();cloud=[]
 for t in q['tasks']:
  if t['id'] not in scope['active_formal_task_ids']:continue
  saved=read(Path(t['tracking_directory'])/'cloud-verification.json');assert saved['verified'] and saved['state']=='FINISHED'
  run=api.run(t['swanlab']['path']);s=run.summary(keys=['test/completed']);assert run.state=='FINISHED' and s['test/completed']['value']==1
  cloud.append({'task':t['id'],'kind':'prediction','path':t['swanlab']['path'],'state':run.state,'summary':s})
 for item in index:
  run=api.run(item['swanlab']['path']);s=run.summary(keys=['quality/completed','quality/dense','quality/method','quality/delta']);assert run.state=='FINISHED' and s['quality/completed']['value']==1
  for k,key in [('quality/dense','dense_value'),('quality/method','method_value'),('quality/delta','delta_method_minus_dense')]:assert abs(s[k]['value']-item['metric'][key])<1e-12
  cloud.append({'task':item['task']['id'],'kind':'quality','path':item['swanlab']['path'],'state':run.state,'summary':s})
 native_proofs=[]
 for encoder_name in ('videomae','timesformer'):
  native_path=ROOT/('classic-'+encoder_name+'-native-result.json');assert sha(native_path)==q['native_gates'][encoder_name]['receipt_sha256']
  native=audit_native_receipt(read(native_path),encoder_name)
  native.update(receipt_path=str(native_path),receipt_sha256=sha(native_path),native_swanlab_url=q['native_gates'][encoder_name]['swanlab_url'])
  native_slug=native['native_swanlab_url'].rstrip('/').split('/')[-1];native_run=api.run('ryfrcd693/VAD-PreBlock-VMA-TS-20261015/'+native_slug);ns=native_run.summary(keys=['qa/completed']);assert native_run.state=='FINISHED' and ns['qa/completed']['value']==1
  native['cloud_state']=native_run.state;native['cloud_summary']=ns;native_proofs.append(native)
 training_proof_path=ROOT/'binding-audit.json';training_proofs=read(training_proof_path)['training'];assert len(training_proofs)==4
 for encoder_name,datasets in c['datasets'].items():
  for dataset_name,dataset in datasets.items():
   proof=training_proofs[encoder_name+('_ucf' if dataset_name=='ucf_crime' else '_xd')]
   assert proof['qa_status']=='passed' and proof['result_status']=='completed' and proof['checkpoint_match'] and proof['metadata_training_view']['available_videos']==dataset['training_videos']==(1610 if dataset_name=='ucf_crime' else 3950)
   contract_path,relocation=historical_contract_location(dataset['training_contract'])
   bindings[contract_path]=dataset['training_contract']['sha256']
   if relocation is not None:historical_contract_relocations[dataset['training_contract']['path']]=relocation
   bindings[dataset['head_run']+'/training_qa.json']=proof['qa_sha256']
 original_eff=api.run(eff['original_cloud_verification']['path']);es=original_eff.summary(keys=['efficiency/completed','efficiency/encoder_groups_completed','efficiency/pipeline_groups_completed']);assert original_eff.state=='FINISHED'
 for k,v in [('efficiency/completed',1),('efficiency/encoder_groups_completed',228),('efficiency/pipeline_groups_completed',228)]:assert es[k]['value']==v
 # No score/model calculation below: exact lookup of already audited measurements.
 encoder={(x['encoder'],x['method'],x['pre_block'],x['remove_percent'],x['batch_size']):x for x in eff['modes']['encoder']['rows']}
 pipeline={(x['encoder'],x['dataset'],x['method'],x['pre_block'],x['remove_percent']):x for x in eff['modes']['pipeline']['rows']}
 rows=[]
 for item in index:
  t=item['task'];m=item['metric'];d=results[t['id']];remove=round(100*(1-t['keep_ratio']));key=(t['encoder'],t['method'],t['target_block'],remove);a=encoder[(*key,1)];b=encoder[(*key,8)];v=pipeline[(t['encoder'],t['dataset'],t['method'],t['target_block'],remove)]
  rows.append({'encoder':t['encoder'],'dataset':t['dataset'],'method':t['method'],'pre_block':t['target_block'],'remove_percent':remove,'metric':m['metric'],'dense_score':m['dense_value'],'method_score':m['method_value'],'delta':m['delta_method_minus_dense'],'ci_low':m['ci_low'],'ci_high':m['ci_high'],'noninferiority':m['noninferiority'],'dense_step_AP':d['frame_ap_step']['dense'],'method_step_AP':d['frame_ap_step']['method'],'encoder_B1_speedup':a['paired_wall_speedup_median_of3'],'encoder_B8_speedup':b['paired_wall_speedup_median_of3'],'pipeline_speedup':v['paired_total_wall_speedup_median_of3'],'pipeline_peak_allocated_MiB':v['peak_allocated_bytes_max_of3']/2**20,'head_sha256':c['datasets'][t['encoder']][t['dataset']]['head_checkpoint']['sha256'],'quality_result_sha256':item['sha256'],'quality_swanlab_url':item['swanlab']['url']})
 rows.sort(key=lambda x:(x['encoder'],x['dataset'],x['method'],x['pre_block'],x['remove_percent']))
 resume_kwargs={}
 if original_report_link is not None:
  prior=api.run(original_report_link['path']);assert prior.name==original_report_link['name'] and prior.summary(keys=['report/completed']).get('report/completed',{}).get('value')!=1
  response=prior._ctx.client.post('/project/ryfrcd693/VAD-PreBlock-VMA-TS-20261015/experiment',{'name':prior.name,'cuid':prior.run_id},retries=0)
  assert response.raw.status_code==200 and response.data['cuid']==prior.run_id and response.data.get('slug')==original_report_link['run_id']
  resume_kwargs={'id':original_report_link['run_id'],'resume':'must'}
 run=swanlab.init(project='VAD-PreBlock-VMA-TS-20261015',workspace='ryfrcd693',name='Active52-final-report-Attention-VisionZip',mode='online',public=False,job_type='final-report-audit',group='user-revised-active52',log_dir=str(args.output/'swanlog'),config={'scope_sha256':SCOPE_SHA,'scientific_campaign_sha256':CAMPAIGN_SHA,'head_training':False,'recompute':False,'formal_cells':52,'quality_comparisons':48,'active_encoder_groups':156,'active_pipeline_groups':156,'uniform_cancelled_by_explicit_user_request':24,'historical_training_contract_relocations':historical_contract_relocations,'original_report_resume':original_report_link is not None},**resume_kwargs)
 link={'url':run.url,'path':run.path.strip('/'),'run_id':run.id,'name':run.name};
 if original_report_link is not None:assert link==original_report_link
 (args.output/'swanlab.json').write_text(json.dumps(link,indent=2))
 try:
  run.save(Path(__file__),base_path=ROOT,policy='now')
  reconciliation=ROOT/'final-report-source-audit-reconciliation.json'
  if original_report_link is not None:
   (args.output/'source-audit-reconciliation.json').write_bytes(reconciliation.read_bytes())
   run.save(reconciliation,base_path=ROOT,policy='now')
  for evidence in [ROOT/'classic-videomae-native-result.json',ROOT/'classic-timesformer-native-result.json',training_proof_path]:run.save(evidence,base_path=ROOT,policy='now')
  remote_runs=[{'id':t['id'],'remote_run':t['remote_run'],'swanlab_url':t['swanlab']['url']} for t in q['tasks'] if t['id'] in scope['active_formal_task_ids']]
  assert all(x['remote_run'].startswith(c['control_root']+'/') and Path(x['remote_run']).name==x['id'] for x in remote_runs)
  remote=verify_remote_bindings(bindings,aliases,remote_runs,[c['output_root']+'/'+ident for ident in scope['excluded_task_ids']])
  payload={'status':'all_active52_computed_audited_original_cloud_verified','generated_hkt':datetime.now(timezone(timedelta(hours=8))).isoformat(),'scope':scope,'scientific_campaign_sha256':CAMPAIGN_SHA,'rows':rows,'quality_source_files':[{'path':x['path'],'sha256':x['sha256']} for x in index],'remote_source_file_audit':remote,'native_bridge_gates':native_proofs,'historical_training_contract_relocations':historical_contract_relocations,'full_training_head_binding_audit':{'source_sha256':sha(training_proof_path),'heads':training_proofs},'all_source_sha256':bindings,'canonical_aliases_revalidated':aliases,'original_cloud_verifications':cloud,'original_efficiency_cloud':{'path':original_eff._proj_path+'/'+original_eff._run_slug,'state':original_eff.state,'summary':es},'efficiency_artifact_sha256':sha(eff_path),'report_swanlab':link,'report_cloud_verification_file':'cloud-verification.json'}
  (args.output/'final-results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
  with (args.output/'quality-efficiency.csv').open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  lines=['# VideoMAE / TimeSformer 活动52格最终报告','','用户在查看旧VideoMAEv2/CLIP正式结果后明确取消后续uniform：48项Attention/VisionZip压缩＋4dense；48项配对质量；活动156 encoder＋156完整视频效率组。此为后验范围变更，不是原完整三方法预注册矩阵；uniform历史保留，不把取消记为完成。','','固定对应完整训练集head、direct_insert，不重训；pre-block6/9/12，信号来自同次前向前一完整block。','','## 架构、预算与固定检测头','','| Encoder | 实际blocks | 每clip输入帧 | 原始tokens | 移除50%后 | 移除70%后 | 预算单位 |','|---|---:|---:|---:|---:|---:|---|','| VideoMAE | 12 | 16 | 1568 | 784 | 470 | tubelet patch |','| TimeSformer | 12 | 8 | 1569 | 785 | 473 | 196条空间轨迹保留98/59条，8帧及原CLS保留 |','','pre6/9/12均在前一完整block5/8/11计算结束后、目标block的LN/attention之前；target block的attention和MLP接收真实短序列。比例按预定整数预算取整。对应adapter readout保持原合同（mean），不把保留CLS偷换成只读CLS。','','每encoder/dataset独立固定UR-DMU检测头，UCF1610个完整训练视频、XD3950个完整可用训练视频；复用原seed0 head，不是64+64 screen head，不重新训练或跨数据集共享head。','','## 全部质量与效率','','UCF主ROC-AUC；XD主梯形PR-AUC。分数单位%，Δ及CI为百分点；倍率小于1代表更慢。','','| Encoder | 数据集 | 方法 | pre层 | 移除% | Dense | 压缩 | Δ pp | 95%CI pp | 非劣 | encoder B8倍率 | 完整视频倍率 |','|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|']
  for x in rows:lines.append(f"| {x['encoder']} | {x['dataset']} | {x['method']} | {x['pre_block']} | {x['remove_percent']} | {100*x['dense_score']:.4f} | {100*x['method_score']:.4f} | {100*x['delta']:+.4f} | [{100*x['ci_low']:+.4f},{100*x['ci_high']:+.4f}] | {x['noninferiority']} | {x['encoder_B8_speedup']:.4f} | {x['pipeline_speedup']:.4f} |")
  lines+=['','## XD step AP补充（不另算CI、不替换主指标）','','| Encoder | 配置 | dense stepAP% | method stepAP% |','|---|---|---:|---:|']
  for x in rows:
   if x['dataset']=='xd_violence':lines.append(f"| {x['encoder']} | {x['method']} pre{x['pre_block']} remove{x['remove_percent']} | {100*x['dense_step_AP']:.4f} | {100*x['method_step_AP']:.4f} |")
  lines+=['','## 解释边界','','- UCF历史训练契约原工作目录已移位，原SHA不变；最终审核使用保留项目中同SHA原契约副本，原路径及副本路径完整记录在JSON，未重建契约或修改协议身份。','- 单head seed0；视频级分层配对bootstrap seed0/10000，非劣margin0.5pp。CI下界须≥−0.5pp才支持非劣；点值小损失不是证明。','- 多配置未整体校正；没有方法间直接配对显著性，不依据test分数选位置/比例/权重。','- 净encoder含adapter/readout/plugin、resident输入；pipeline含16完整训练视频原native CPU准备/H2D/encoder/D2H/固定head/frame projection，不含FeatureStore落盘、权重加载与源SHA预查。','- Encoder-only计时使用固定UCF训练正常视频Normal_Videos001_x264输入，B1/B8分别测量；同encoder的XD表行复用该encoder时序，并非另测XD encoder。完整视频pipeline则按UCF/XD各自16个完整训练视频分别测量。','- 效率为三独立进程配对中位倍率；GPU独占，CPU/存储共享、OS页缓存未控制；倍率不是速度显著性。CPU生产优化吞吐单列，不替换原native基准。','- 原所有228+228效率组与uniform历史完整保留；本报告仅按明确用户范围保留活动156+156，保留两方法全部原固定位置和预算，不按分数只报告最佳格；uniform排除发生于用户查看历史模型正式结果之后，必须披露为后验范围变更，不能据此推断新encoder上的uniform质量。','- 最终报告上传终态以本地cloud-verification.json为准，只有verified=true才算整套交付完成。']
  (args.output/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
  for name in ['encoder-efficiency-active.csv','pipeline-efficiency-active.csv']: (args.output/name).write_bytes((eff_path.parent/name).read_bytes())
  manifest={x.name:sha(x) for x in args.output.iterdir() if x.is_file()};(args.output/'artifact-sha256.json').write_text(json.dumps(manifest,indent=2))
  for x in args.output.iterdir():
   if x.is_file():run.save(x.resolve(),base_path=args.output.resolve(),policy='now')
  run.log({'report/formal_completed':52,'report/quality_completed':48,'report/encoder_groups':156,'report/pipeline_groups':156,'report/completed':1},step=1);run.finish()
 except Exception as exc:
  run.finish(state='crashed',error=str(exc));raise
 fresh=swanlab.Api().run(link['path']);s=fresh.summary(keys=['report/completed','report/formal_completed','report/quality_completed']);verified=fresh.state=='FINISHED' and all(s.get(k,{}).get('value')==v for k,v in [('report/completed',1),('report/formal_completed',52),('report/quality_completed',48)])
 (args.output/'cloud-verification.json').write_text(json.dumps(dict(link,state=fresh.state,verified=verified,completion_summary=s),indent=2));assert verified,'final report cloud pending; preserve original run, no ML repeat'
 print('FINAL_ACTIVE52_COMPLETE_AND_CLOUD_VERIFIED',link['url'])
if __name__=='__main__':main()

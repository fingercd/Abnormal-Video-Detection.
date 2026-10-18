"""Single local four-slot relay. Never auto-replay an uncertain/failed job."""
import argparse,json,os,subprocess,sys,time,hashlib,psutil
from controller_state_io import atomic_json_write
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--sha',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--execution-ready',type=Path,required=True);p.add_argument('--extra-slot',type=int,action='append',default=[]);p.add_argument('--scope-override',type=Path,required=True);p.add_argument('--scope-sha',required=True);p.add_argument('--efficiency-proof',type=Path);p.add_argument('--efficiency-proof-sha');p.add_argument('--efficiency-audit',type=Path);p.add_argument('--efficiency-audit-sha');p.add_argument('--resource-policy',type=Path,required=True);p.add_argument('--resource-policy-sha',required=True);a=p.parse_args()
policy=json.loads(a.resource_policy.read_text(encoding='utf-8'))
if hashlib.sha256(a.resource_policy.read_bytes()).hexdigest()!=a.resource_policy_sha or policy['allowed_gpu_slots']!=list(range(8)) or not policy['allow_shared_other_processes'] or policy['minimum_free_mib_guard'] is not None:raise ValueError('latest explicit shared-GPU policy binding differs')
if any(g not in (4,5,6,7) for g in a.extra_slot):raise ValueError('only node3 slots0..7 authorized')
if 5 in a.extra_slot:
 if not a.efficiency_proof or not a.efficiency_audit:raise ValueError('GPU5 still reserved until full original efficiency audit and cloud proof')
 if hashlib.sha256(a.efficiency_proof.read_bytes()).hexdigest()!=a.efficiency_proof_sha or hashlib.sha256(a.efficiency_audit.read_bytes()).hexdigest()!=a.efficiency_audit_sha:raise ValueError('GPU5 release evidence changed')
 proof=json.loads(a.efficiency_proof.read_text());audit=json.loads(a.efficiency_audit.read_text())
 if proof['run_id']!='r4ef7f3u' or not proof['verified'] or proof['state']!='FINISHED' or audit['groups']!=228 or audit['summary_cells']!=76 or audit['status']!='original_native_pipeline_complete_and_cloud_verified':raise ValueError('original456 efficiency release not verified')
 for key in ('efficiency/completed','efficiency/encoder_groups_completed','efficiency/pipeline_groups_completed'):
  if proof['completion_summary'][key]['value']!=(1 if key=='efficiency/completed' else 228):raise ValueError('original efficiency counters incomplete')
if len(set(a.extra_slot))!=len(a.extra_slot):raise ValueError('duplicate extra slot')
c=json.loads(a.campaign.read_text(encoding='utf-8-sig'))
if hashlib.sha256(a.campaign.read_bytes()).hexdigest()!=a.sha:raise ValueError('campaign SHA differs')
scope=json.loads(a.scope_override.read_text(encoding='utf-8'))
if hashlib.sha256(a.scope_override.read_bytes()).hexdigest()!=a.scope_sha or scope['scientific_campaign_sha256']!=a.sha:raise ValueError('explicit user execution scope binding differs')
active_formal={t['id'] for t in c['tasks'] if t['method']!='uniform'}
excluded={t['id'] for t in c['tasks'] if t['method']=='uniform'}
if set(scope['excluded_task_ids'])!=excluded or set(scope['active_formal_task_ids'])!=active_formal or len(active_formal)!=52 or len(excluded)!=24 or scope['reason']!='explicit_user_request_stop_future_uniform':raise ValueError('scope must match exact user-cancelled24 uniform tasks')
r=a.root.resolve();r.mkdir(exist_ok=True);lock=r/'controller.pid'
with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
repo=Path.cwd().resolve();qpath=repo/'work/preblock-vma-ts-20261015-r01/queue.json'
q=json.loads(qpath.read_text(encoding='utf-8-sig'));old_controller_pid=q.get('controller_pid');
if a.resume and old_controller_pid and psutil.pid_exists(old_controller_pid):raise RuntimeError('prior controller PID exists; inspect before another relay starts')
q['controller_recovered_from_pid']=old_controller_pid if a.resume else None;q['controller_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();q['status']='formal_r02_relay_running';q['campaign_sha256']=a.sha;q['campaign_path']=str(a.campaign.resolve());q['expected_snapshot']=c['snapshot_root'];q['controller_pid']=os.getpid()
execution_slots=list(c['gpu_slots'])+a.extra_slot
q['execution_gpu_slots']=execution_slots
q['execution_resource_authorization']=policy;q['resource_policy_sha256']=a.resource_policy_sha
tasks={t['id']:t for t in q['tasks']}
expected={t['id']:t for t in c['tasks']}
if set(tasks)!=set(expected):raise ValueError('queue differs from the exact76 frozen task identities')
for t in tasks.values():
 if t['id'] in excluded and t['status']!='cancelled_user_scope_change':raise ValueError('excluded uniform task is not cancelled')
 if any(t[k]!=expected[t['id']][k] for k in ('encoder','dataset','method','target_block','keep_ratio')):raise ValueError('queue scientific configuration differs')
 if not a.resume:
  if t['status'] not in ('pending','held_after_preflight_failure'):raise ValueError('use resume to adopt original jobs, never replay')
  t['status']='pending'
slots={};children={}
class AdoptedLogger:
 def __init__(self,t):
  self.pid=t['logger_pid'];self.returncode=None;self.track=Path(t['tracking_directory'])
  if t['expected_snapshot']!=c['snapshot_root']:raise ValueError('original expected snapshot differs')
  if Path(t['swanlab_source']).resolve()!=(self.track/'swanlab.json').resolve():raise ValueError('original Swan source pointer differs')
  link=json.loads((self.track/'swanlab.json').read_text());provenance=json.loads((self.track.parent/'provenance.json').read_text())
  if link!=t['swanlab'] or provenance['campaign']!=c or provenance['task']['id']!=t['id']:raise ValueError('original Swan/campaign/task identity differs')
  self.process=psutil.Process(self.pid) if psutil.pid_exists(self.pid) else None
  if self.process is not None:
   args=self.process.cmdline()
   expected_tracker=Path(t.get('tracker_script',str(repo/'scripts/icassp2027/track_relation_qa.py'))).resolve()
   if Path(args[1]).resolve()!=expected_tracker:raise ValueError('logger PID belongs to another command')
   if t.get('tracker_source_sha256') and hashlib.sha256(expected_tracker.read_bytes()).hexdigest()!=t['tracker_source_sha256']:raise ValueError('tracker source changed')
   if Path(args[args.index('--output')+1]).resolve()!=self.track.resolve() or Path(args[args.index('--provenance')+1]).resolve()!=(self.track.parent/'provenance.json').resolve():raise ValueError('logger PID belongs to another original run')
   if abs(self.process.create_time()-datetime.fromisoformat(t.get('logger_attached_utc',t['started_utc'])).timestamp())>5:raise ValueError('logger PID creation time does not match original launch')
 def poll(self):
  if self.process is not None and self.process.is_running():return None
  p=self.track/'cloud-verification.json';v=json.loads(p.read_text()) if p.exists() else {}
  self.returncode=0 if v.get('verified') and v.get('state')=='FINISHED' and v.get('completion_summary',{}).get('test/completed',{}).get('value')==1 else -199
  return self.returncode
if a.resume:
 for t in tasks.values():
  if t['status']=='running':
   gpu=t['gpu_slot']
   if gpu in slots:raise ValueError('two original tasks claim the same GPU slot')
   children[gpu]=AdoptedLogger(t);slots[gpu]=t['id'];print('ADOPT_ORIGINAL',t['id'],gpu,t['logger_pid'],t['swanlab']['run_id'],flush=True)
  elif t['status']=='prediction_completed':
   v=json.loads((Path(t['tracking_directory'])/'cloud-verification.json').read_text())
   if not(v['verified'] and v['state']=='FINISHED' and v['completion_summary']['test/completed']['value']==1):raise ValueError('cannot count an unverified old completion')
share={0:('1827','yangrunze','/users/yangrunze/tcmid-rag-gpu/venv/bin/python -u -m uvicorn gpu_service:app --host 127.0.0.1 --port 8003 --workers 1 --log-level info'),1:('15743','fotile','/users/fotile/CVPR-JY2/.venv/bin/python -u -m runtime.cli train --config /users/fotile/CVPR-JY2-runs/idea3-round3-20261015/configs/third_idea/round3_E25.json --output /users/fotile/CVPR-JY2/outputs/current_idea3_round3_E25_20261015'),2:('15744','fotile','/users/fotile/CVPR-JY2/.venv/bin/python -u -m runtime.cli train --config /users/fotile/CVPR-JY2-runs/idea3-round3-20261015/configs/third_idea/round3_E26.json --output /users/fotile/CVPR-JY2/outputs/current_idea3_round3_E26_20261015')}
def save():
 q['updated_utc']=datetime.now(timezone.utc).isoformat();q['formal_completed']=sum(t['status']=='prediction_completed' for t in tasks.values() if t['id'] in active_formal);q['active_formal_total']=len(active_formal);q['scope_override']=str(a.scope_override.resolve());q['scope_override_sha256']=a.scope_sha;atomic_json_write(qpath,q)
resource_wait_until={}
def gpu_ready(gpu):
 # Shared foreign PIDs and low free VRAM are observations, not launch rejection.
 allowed=share.get(gpu)
 settings={'gpu':gpu,'allowed':allowed}
 code="""import json,subprocess,os,pwd
from pathlib import Path
assert os.uname().nodename=='ibnode3'
s=SETTINGS
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i',str(s['gpu'])],text=True).strip()
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i',str(s['gpu'])],text=True).strip())
lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
unknown=[]
for line in lines:
 device,pid=[x.strip() for x in line.split(',')]
 if device!=uuid:continue
 proc=Path('/proc')/pid;owner=pwd.getpwuid(proc.stat().st_uid).pw_name if proc.exists() else None
 command=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode().strip() if proc.exists() else None
 if s['allowed'] is None or (pid,owner,command)!=tuple(s['allowed']):unknown.append({'pid':pid,'owner':owner,'command':command})
print(json.dumps({'gpu':s['gpu'],'uuid':uuid,'free_mib':free,'unknown':unknown,'ready':True,'sharing_authorized':True}))
""".replace('SETTINGS',repr(settings))
 result=subprocess.run(['ssh','-T','-p','12345','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','fotile@121.196.228.153','python3','-'],input=code.encode(),capture_output=True,timeout=45)
 if result.returncode:raise RuntimeError('pre-online GPU guard probe failed; no new Swan/job launch: '+result.stderr.decode(errors='replace'))
 observed=json.loads(result.stdout)
 print('GPU_SHARED_OBSERVATION',json.dumps(observed),flush=True)
 return True


def launch(t,gpu):
 d=r/'tasks'/t['id']
 run=c['control_root']+'/tasks-execution-r04/'+t['id'];snap=c['snapshot_root'];remote_campaign=c['control_root']+'/campaign-r02.json'
 source_archive=a.campaign.parent/'formal-source-r02.tar.gz'
 override=None
 executable=f"{snap}/scripts/icassp2027/run_pre_block_test.py"
 if t['encoder'] in ('videomae','timesformer') and t['dataset'] in ('ucf_crime','xd_violence'):
  override=json.loads(a.execution_ready.read_text(encoding='utf-8'))['configs_by_pair'][t['encoder']+'-'+t['dataset']]
  if override['status']!='ready' or override['scientific_campaign_sha256']!=a.sha:raise ValueError('unapproved or different scientific campaign')
  for proof in override['local_bindings'].values():
   if hashlib.sha256(Path(proof['path']).read_bytes()).hexdigest()!=proof['sha256']:raise ValueError('execution proof changed')
  acceptance=json.loads(Path(override['local_bindings']['acceptance']['path']).read_text())
  cloud=json.loads(Path(override['local_bindings']['cloud_verification']['path']).read_text())
  if acceptance['status']!='passed' or acceptance['candidate_sha256']!=override['candidate_sha256'] or not cloud['verified'] or cloud['state']!='FINISHED' or cloud['completion_summary']['qa/completed']['value']!=1:raise ValueError('exactness or cloud gate not passed')
  source_archive=Path(override['source_bundle'])
  if hashlib.sha256(source_archive.read_bytes()).hexdigest()!=override['source_bundle_sha256']:raise ValueError('execution source bundle changed')
  executable=f"{override['remote_launcher']} --execution-config {override['remote_config']} --execution-config-sha256 {override['remote_config_sha256']} --script {executable}"
 d.mkdir(parents=True,exist_ok=False)
 guard='echo "SHARED_GPU_USER_AUTHORIZED existing_PIDs=$pids"'
 script=f'''set -euo pipefail
[ "$(hostname)" = ibnode3 ]
RUN={run}
[ ! -e "$RUN" ];mkdir -p "$RUN"
printf '%s\\n' __SWANLAB_URL__ > "$RUN/swanlab-url.txt"
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i {gpu})
pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v uuid="$uuid" '$1 == uuid {{print $2}}')
{guard}
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i {gpu})
echo "GPU_SLOT={gpu} USER_SHARED_START_FREE_MIB=$free"
cat > "$RUN/run.sh" <<'RUNNER'
set +e
export CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH={snap}/src
/users/fotile/VAD/.encoder-envs/v2/classic-video-v2/bin/python -u {executable} --campaign {remote_campaign} --campaign-sha256 {a.sha} --task {t['id']} --tracking-url "$VAD_TRACKING_URL"
rc=$?;echo "$rc" > {run}/exit-code.txt;exit "$rc"
RUNNER
export VAD_TRACKING_URL=$(cat "$RUN/swanlab-url.txt")
nohup bash "$RUN/run.sh" > "$RUN/task.log" 2>&1 < /dev/null &
pid=$!;echo "$pid" > "$RUN/pid.txt"
echo "FORMAL_PID=$pid GPU_SLOT={gpu} UUID=$uuid"
tail --pid="$pid" -n +1 -f "$RUN/task.log"
exit "$(cat "$RUN/exit-code.txt")"
'''
 (d/'launch.sh').write_text(script,encoding='utf-8');(d/'provenance.json').write_text(json.dumps({'phase':'preblock_formal_test','remote_run':run,'campaign':c,'task':{k:t[k] for k in ('id','encoder','dataset','method','target_block','keep_ratio')},'gpu_slot':gpu,'expected_snapshot':snap,'source_archive_sha256':hashlib.sha256(source_archive.read_bytes()).hexdigest(),'scientific_source_archive_sha256':c['source_archive_sha256'],'execution_override':override,'seed':0,'attempt':'r02','head_training':False,'protected_shared_process':share.get(gpu),'execution_gpu_slots':execution_slots,'shared_gpu_policy':policy,'shared_gpu_policy_sha256':a.resource_policy_sha},indent=2),encoding='utf-8')
 command=[sys.executable,str(a.campaign.parent/'track_preblock_online.py'),'--launch-script',str(d/'launch.sh'),'--provenance',str(d/'provenance.json'),'--source-archive',str(source_archive),'--output',str(d/'tracking'),'--name',t['id']+'-r02','--project','VAD-PreBlock-VMA-TS-20261015','--group','frozen76-classic-r02','--completion-key','test/completed','--failure-key','test/failed','--job-type','formal-test-extraction-prediction','--host','fotile@121.196.228.153','--port','12345']
 if t.get('reserved_swanlab_source'):
  wrapper=Path(t.get('first_launch_wrapper',str(a.campaign.parent/'resume_original_first_launch.py')))
  original_link=json.loads(Path(t['reserved_swanlab_source']).read_text())
  if not t.get('prelaunch_no_gpu_started') or original_link!=t['swanlab']:raise ValueError('original first-execution identity not proven')
  command=[sys.executable,str(wrapper),'--original-swanlab',t['reserved_swanlab_source'],'--frozen-tracker',str(a.campaign.parent/'track_preblock_online.py'),*command[2:]]
  provenance=json.loads((d/'provenance.json').read_text());provenance['original_first_execution_resume']={'original_swanlab':original_link,'wrapper_sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest(),'previous_no_remote_run_receipt':t['remote_absence_receipt']};(d/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
 with (d/'logger.stdout').open('w',encoding='utf-8') as out,(d/'logger.stderr').open('w',encoding='utf-8') as err:
  child=subprocess.Popen(command,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 (d/'logger.pid').write_text(str(child.pid));children[gpu]=child;slots[gpu]=t['id']
 t.update(status='running',gpu_slot=gpu,logger_pid=child.pid,tracking_directory=str(d/'tracking'),swanlab_source=str(d/'tracking'/'swanlab.json'),expected_snapshot=snap,execution_override=override,tracker_script=command[1],tracker_source_sha256=hashlib.sha256(Path(command[1]).read_bytes()).hexdigest(),remote_run=run,attempt='r02',started_utc=datetime.now(timezone.utc).isoformat());print('LAUNCH',t['id'],gpu,child.pid,flush=True)
try:
 while True:
  for gpu in list(children):
   child=children[gpu];ident=slots[gpu];t=tasks[ident];track=Path(t['tracking_directory'])
   if (track/'swanlab.json').exists():t['swanlab']=json.loads((track/'swanlab.json').read_text())
   if child.poll() is None:continue
   receipt=json.loads((track/'cloud-verification.json').read_text()) if (track/'cloud-verification.json').exists() else {}
   ok=child.returncode==0 and receipt.get('verified') and receipt.get('state')=='FINISHED' and receipt.get('completion_summary',{}).get('test/completed',{}).get('value')==1
   t['status']='prediction_completed' if ok else 'held_inspect_original_remote_pid';t['logger_exit']=child.returncode;t['cloud_verification']=receipt;t['finished_utc']=datetime.now(timezone.utc).isoformat();print('TERMINAL',ident,t['status'],child.returncode,flush=True)
   del children[gpu];del slots[gpu]
  # A held failure blocks only its slot; never retry automatically or pile up
  # attempts when transport state or compatibility is uncertain.
  held_gpus={t.get('gpu_slot') for t in tasks.values() if t['status']=='held_inspect_original_remote_pid'}
  for gpu in execution_slots:
   if gpu not in slots and gpu not in held_gpus:
    pending=[tasks[x['id']] for x in c['tasks'] if tasks[x['id']]['status']=='pending']
    preferred_encoder='timesformer' if gpu in (5,7) else ('videomae' if gpu in (4,6) else None)
    next_task=next((t for t in pending if t['encoder']==preferred_encoder),pending[0] if pending else None)
    if next_task is not None and gpu_ready(gpu):launch(next_task,gpu)
  save()
  if not children:
   q['status']='all_predictions_completed' if q['formal_completed']==len(active_formal) else 'relay_held_original_pid_inspection_required';save();break
  time.sleep(20)
finally:
 save()

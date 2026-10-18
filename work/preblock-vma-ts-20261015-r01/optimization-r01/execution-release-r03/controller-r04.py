"""Single local four-slot relay. Never auto-replay an uncertain/failed job."""
import argparse,json,os,subprocess,sys,time,hashlib,psutil
from controller_state_io import atomic_json_write
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--sha',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--execution-ready',type=Path,required=True);a=p.parse_args()
c=json.loads(a.campaign.read_text(encoding='utf-8-sig'))
if hashlib.sha256(a.campaign.read_bytes()).hexdigest()!=a.sha:raise ValueError('campaign SHA differs')
r=a.root.resolve();r.mkdir(exist_ok=True);lock=r/'controller.pid'
with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
repo=Path.cwd().resolve();qpath=repo/'work/preblock-vma-ts-20261015-r01/queue.json'
q=json.loads(qpath.read_text(encoding='utf-8-sig'));old_controller_pid=q.get('controller_pid');
if a.resume and old_controller_pid and psutil.pid_exists(old_controller_pid):raise RuntimeError('prior controller PID exists; inspect before another relay starts')
q['controller_recovered_from_pid']=old_controller_pid if a.resume else None;q['controller_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();q['status']='formal_r02_relay_running';q['campaign_sha256']=a.sha;q['campaign_path']=str(a.campaign.resolve());q['expected_snapshot']=c['snapshot_root'];q['controller_pid']=os.getpid()
tasks={t['id']:t for t in q['tasks']}
expected={t['id']:t for t in c['tasks']}
if set(tasks)!=set(expected):raise ValueError('queue differs from the exact76 frozen task identities')
for t in tasks.values():
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
   if Path(args[1]).resolve()!=repo/'scripts/icassp2027/track_relation_qa.py':raise ValueError('logger PID belongs to another command')
   if Path(args[args.index('--output')+1]).resolve()!=self.track.resolve() or Path(args[args.index('--provenance')+1]).resolve()!=(self.track.parent/'provenance.json').resolve():raise ValueError('logger PID belongs to another original run')
   if abs(self.process.create_time()-datetime.fromisoformat(t['started_utc']).timestamp())>5:raise ValueError('logger PID creation time does not match original launch')
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
 q['updated_utc']=datetime.now(timezone.utc).isoformat();q['formal_completed']=sum(t['status']=='prediction_completed' for t in tasks.values());atomic_json_write(qpath,q)
def launch(t,gpu):
 d=r/'tasks'/t['id']
 run=c['control_root']+'/tasks-execution-r03/'+t['id'];snap=c['snapshot_root'];remote_campaign=c['control_root']+'/campaign-r02.json'
 source_archive=a.campaign.parent/'formal-source-r02.tar.gz'
 override=None
 executable=f"{snap}/scripts/icassp2027/run_pre_block_test.py"
 if (t['encoder'],t['dataset'])==('videomae','xd_violence'):
  override=json.loads(a.execution_ready.read_text(encoding='utf-8'))
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
 if gpu in share:
  pid,owner,command=share[gpu]
  import shlex
  guard=f'''for pid in $pids; do
[ "$pid" = "{pid}" ] || {{ echo "Unknown process; no launch";exit 22; }}
[ "$(ps -p "$pid" -o user=)" = "{owner}" ]
command=$(tr '\\0' ' ' < "/proc/$pid/cmdline")
[ "$command" = {shlex.quote(command+' ')} ] || {{ echo "Protected process identity changed; no launch";exit 22; }}
done'''
 else:guard='[ -z "$pids" ] || { echo "Unknown GPU process; no launch";exit 22; }'
 script=f'''set -euo pipefail
[ "$(hostname)" = ibnode3 ]
RUN={run}
[ ! -e "$RUN" ];mkdir -p "$RUN"
printf '%s\\n' __SWANLAB_URL__ > "$RUN/swanlab-url.txt"
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i {gpu})
pids=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | awk -F', ' -v uuid="$uuid" '$1 == uuid {{print $2}}')
{guard}
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i {gpu})
[ "$free" -ge 24576 ] || {{ echo "Insufficient free VRAM; no launch";exit 22; }}
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
 (d/'launch.sh').write_text(script,encoding='utf-8');(d/'provenance.json').write_text(json.dumps({'phase':'preblock_formal_test','campaign':c,'task':{k:t[k] for k in ('id','encoder','dataset','method','target_block','keep_ratio')},'gpu_slot':gpu,'expected_snapshot':snap,'source_archive_sha256':hashlib.sha256(source_archive.read_bytes()).hexdigest(),'scientific_source_archive_sha256':c['source_archive_sha256'],'execution_override':override,'seed':0,'attempt':'r02','head_training':False,'protected_shared_process':share.get(gpu)},indent=2),encoding='utf-8')
 command=[sys.executable,str(repo/'scripts/icassp2027/track_relation_qa.py'),'--launch-script',str(d/'launch.sh'),'--provenance',str(d/'provenance.json'),'--source-archive',str(source_archive),'--output',str(d/'tracking'),'--name',t['id']+'-r02','--project','VAD-PreBlock-VMA-TS-20261015','--group','frozen76-classic-r02','--completion-key','test/completed','--failure-key','test/failed','--job-type','formal-test-extraction-prediction']
 with (d/'logger.stdout').open('w',encoding='utf-8') as out,(d/'logger.stderr').open('w',encoding='utf-8') as err:
  child=subprocess.Popen(command,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 (d/'logger.pid').write_text(str(child.pid));children[gpu]=child;slots[gpu]=t['id']
 t.update(status='running',gpu_slot=gpu,logger_pid=child.pid,tracking_directory=str(d/'tracking'),swanlab_source=str(d/'tracking'/'swanlab.json'),expected_snapshot=snap,execution_override=override,remote_run=run,attempt='r02',started_utc=datetime.now(timezone.utc).isoformat());print('LAUNCH',t['id'],gpu,child.pid,flush=True)
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
  for gpu in c['gpu_slots']:
   if gpu not in slots and gpu not in held_gpus:
    next_task=next((tasks[x['id']] for x in c['tasks'] if tasks[x['id']]['status']=='pending'),None)
    if next_task is not None:launch(next_task,gpu)
  save()
  if not children:
   q['status']='all_predictions_completed' if q['formal_completed']==76 else 'relay_held_original_pid_inspection_required';save();break
  time.sleep(20)
finally:
 save()

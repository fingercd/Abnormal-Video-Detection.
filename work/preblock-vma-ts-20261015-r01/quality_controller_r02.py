"""Resume the frozen one-worker CPU-quality queue without repeating comparisons.

Observer recovery only: completed original SwanLab/results are reused; uncertain
original PID/run identities hold the queue. New work uses the unchanged frozen
comparison entry point and the reviewed short-SSH online tracker.
"""
import argparse,ast,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
import psutil
from controller_state_io import atomic_json_write

PROJECT='VAD-PreBlock-VMA-TS-20261015'
STATIC=('id','encoder','dataset','method','target_block','keep_ratio')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def source_link(row):
    source=Path(row['swanlab_source']);source=source/'swanlab.json' if source.is_dir() else source
    link=read(source);track=Path(row['tracking_directory'])
    if read(track/'swanlab.json')!=link or (row.get('swanlab') is not None and row['swanlab']!=link):
        raise ValueError('original quality SwanLab identities differ')
    return link


def tracker_helper(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)
             and any(a.name=='stream_original_task' for a in n.names)]
    if len(modules)!=1 or modules[0] not in ('short_poll_original','short_poll_original_r02'):
        raise ValueError('quality tracker must import the reviewed local short-poll helper')
    return Path(path).parent/(modules[0]+'.py')


def validate_tracker(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    constants={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    assignments=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='polling' for t in n.targets)]
    if not {'--formal-quality','--host','--port'}<=constants:
        raise ValueError('tracker lacks explicit formal-quality/host/port support')
    if not any('preblock_quality' in {n.value for n in ast.walk(value) if isinstance(n,ast.Constant)} for value in assignments):
        raise ValueError('tracker has no short-SSH preblock_quality mode; never mislabel CPU quality as formal test')
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='stream_original_task']
    if not any(any(isinstance(n,ast.Attribute) and n.attr=='completion_key' for n in ast.walk(call)) for call in calls):
        raise ValueError('tracker must forward quality completion_key to short polling')
    helper=ast.parse(tracker_helper(path).read_text(encoding='utf-8-sig'))
    if not any(isinstance(n,ast.FunctionDef) and n.name=='stream_original_task'
               and any(arg.arg=='completion_key' for arg in [*n.args.args,*n.args.kwonlyargs]) for n in ast.walk(helper)):
        raise ValueError('short polling helper is hardcoded to test completion, not quality')


def validate_rows(state,c,expected_sha):
    if state['campaign_sha256']!=expected_sha:raise ValueError('original quality campaign SHA differs')
    expected={t['id']:{k:t[k] for k in STATIC} for t in c['tasks'] if t['method']!='identity'}
    actual={row['task']['id']:{k:row['task'][k] for k in STATIC} for row in state['jobs']}
    if len(state['jobs'])!=72 or len(actual)!=72 or actual!=expected:
        raise ValueError('original quality queue is not the exact frozen 72 comparisons')
    if c['bootstrap_seed']!=0 or c['bootstrap_draws']!=10000:
        raise ValueError('quality recovery cannot change the frozen bootstrap')


def completion_evidence(row,c,cached_root,remote_checker=None):
    track=Path(row['tracking_directory']);link=source_link(row)
    expected=c['control_root']+'/efficiency-source-r01'
    if row['expected_snapshot']!=expected:raise ValueError('original quality snapshot differs from frozen comparator')
    receipt=read(track/'cloud-verification.json')
    if (any(receipt.get(k)!=link[k] for k in ('url','path','run_id')) or not receipt.get('verified')
            or receipt.get('state')!='FINISHED'
            or receipt.get('completion_summary',{}).get('quality/completed',{}).get('value')!=1):
        raise ValueError('original quality has no verified FINISHED completion')
    events=[json.loads(line) for line in (track/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    complete=[e for e in events if e.get('task_id')==row['task']['id'] and e.get('metrics',{}).get('quality/completed')==1]
    if len(complete)!=1:raise ValueError('original quality completion event is absent/ambiguous')
    event=complete[0];result=event['quality'];metric=result['metric']
    if (result['dataset']!=row['task']['dataset'] or result['tracking_url']!=link['url']
            or any(result['campaign_task'][key]!=row['task'][key] for key in STATIC)):
        raise ValueError('original quality result task/dataset/tracking identity differs')
    if metric['bootstrap_seed']!=c['bootstrap_seed'] or metric['bootstrap_draws']!=c['bootstrap_draws']:
        raise ValueError('original result bootstrap identity differs')
    expected_result=str(Path(c['output_root'])/row['task']['id']/'quality-preview.json').replace('\\','/')
    if event['quality_path']!=expected_result:raise ValueError('original quality result path differs')
    cached=Path(cached_root)/(row['task']['id']+'.json')
    if cached.exists():
        if sha(cached)!=event['quality_sha256'] or read(cached)!=result:
            raise ValueError('original cached quality result bytes/payload differ')
    elif remote_checker is None or remote_checker(row).get('quality_sha256')!=event['quality_sha256']:
        raise ValueError('original quality bytes are not verified; never recompute')
    return receipt


def load_scope_override(path,expected_sha,c,campaign_sha):
    if path is None and expected_sha is None:return None
    if path is None or expected_sha is None:raise ValueError('scope override requires both path and SHA')
    if sha(path)!=expected_sha:raise ValueError('scope override SHA differs')
    value=read(path)
    tasks={t['id']:t for t in c['tasks']}
    excluded={ident for ident,t in tasks.items() if t['method']=='uniform'}
    formal=set(tasks)-excluded
    quality={ident for ident in formal if tasks[ident]['method']!='identity'}
    if (len(c['tasks'])!=76 or len(tasks)!=76 or len(excluded)!=24 or len(formal)!=52 or len(quality)!=48
            or value['scientific_campaign_sha256']!=campaign_sha
            or value['reason']!='explicit_user_request_stop_future_uniform'):
        raise ValueError('scope override is not the explicit frozen no-uniform change')
    for key,expected in (('excluded_task_ids',excluded),('active_formal_task_ids',formal),('active_quality_task_ids',quality)):
        ids=value[key]
        if not isinstance(ids,list) or len(ids)!=len(set(ids)) or set(ids)!=expected:
            raise ValueError('scope override '+key+' differs from exact frozen method/task set')
    return dict(value,path=str(Path(path).resolve()),sha256=expected_sha)


def apply_scope_override(state,scope):
    if scope is None:
        if state.get('scope_override') or any(row['status']=='cancelled_user_scope_change' for row in state['jobs']):
            raise ValueError('original scope cancellation requires the original bound override')
        return {row['task']['id'] for row in state['jobs']}
    if state.get('scope_override') and (state.get('scope_override_sha256')!=scope['sha256']
            or Path(state['scope_override']).resolve()!=Path(scope['path']).resolve()):
        raise ValueError('original user scope override binding differs')
    excluded=set(scope['excluded_task_ids'])
    for row in state['jobs']:
        if row['status']=='cancelled_user_scope_change' and row['task']['id'] not in excluded:
            raise ValueError('non-uniform task cannot be cancelled by this scope')
        if row['task']['id'] in excluded and row['status']=='pending':
            row['status']='cancelled_user_scope_change'
            row['cancellation_reason']=scope['reason']
            row['scope_override_sha256']=scope['sha256']
    state['scope_override']=scope['path']
    state['scope_override_sha256']=scope['sha256']
    state['scope_override_details']=scope
    return set(scope['active_quality_task_ids'])


def quality_counts(state,active_ids):
    return {'completed':sum(row['status']=='completed' for row in state['jobs']),
            'historical_total':len(state['jobs']),'active_quality_total':len(active_ids),
            'active_completed':sum(row['status']=='completed' and row['task']['id'] in active_ids for row in state['jobs']),
            'cancelled_user_scope_change':sum(row['status']=='cancelled_user_scope_change' for row in state['jobs'])}


def next_ready(state,formal,active_ids=None):
    if any(row['status'] not in ('completed','pending','cancelled_user_scope_change') for row in state['jobs']):return None
    active_ids=set(row['task']['id'] for row in state['jobs']) if active_ids is None else active_ids
    done={t['id'] for t in formal['tasks'] if t['status']=='prediction_completed'}
    return next((row for row in state['jobs'] if row['status']=='pending' and row['task']['id'] in active_ids and row['task']['id'] in done
                 and row['task']['encoder']+'-'+row['task']['dataset']+'-dense' in done),None)


def ssh_probe(row,c,*,pending=False,host='fotile@121.196.228.153',port=12345):
    task=row['task'];run=row.get('remote_run',c['control_root']+'/quality-r01/'+task['id'])
    target=str(Path(c['output_root'])/task['id']/'quality-preview.json').replace('\\','/')
    settings={'run':run,'quality':target,'pending':pending,'url':None if pending else source_link(row)['url'],'snapshot':row.get('expected_snapshot')}
    code="""import json,hashlib,pwd\nfrom pathlib import Path\ns=SETTINGS\nr=Path(s['run']);q=Path(s['quality'])\nif not s['pending'] and (r/'swanlab-url.txt').read_text().strip()!=s['url']:raise ValueError('original remote SwanLab URL differs')\nif not s['pending'] and s['snapshot'] not in (r/'run.sh').read_text():raise ValueError('original remote quality snapshot differs')\nv={'remote_run_exists':r.exists(),'quality_exists':q.exists()}\nif q.exists():v['quality_sha256']=hashlib.sha256(q.read_bytes()).hexdigest()\nif (r/'pid.txt').exists():\n pid=int((r/'pid.txt').read_text());p=Path('/proc')/str(pid);v['pid']=pid;v['live']=p.exists()\n if p.exists():\n  command=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode().strip();owner=pwd.getpwuid(p.stat().st_uid).pw_name\n  if owner!='fotile' or str(r/'run.sh') not in command:raise ValueError('original CPU PID owner/command differs')\n  v.update(command=command,owner=owner,start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19])\nelse:v['live']=False\nif (r/'exit-code.txt').exists():v['exit_code']=int((r/'exit-code.txt').read_text())\nprint(json.dumps(v))\n""".replace('SETTINGS',repr(settings))
    command=['ssh','-T','-p',str(port),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',host,'python3','-']
    for attempt in range(3):
        try:result=subprocess.run(command,input=code.encode(),capture_output=True,timeout=35)
        except subprocess.TimeoutExpired as exc:error=exc
        else:
            if result.returncode==0:return json.loads(result.stdout)
            message=result.stderr.decode(errors='replace')
            if result.returncode!=255 or any(v in message.lower() for v in ('permission denied','host key verification failed','identification has changed')):
                raise RuntimeError('read-only original CPU probe failed: '+message)
            error=RuntimeError(message)
        if attempt<2:time.sleep(3*(attempt+1))
    raise RuntimeError('short SSH probe transport exhausted; hold original CPU identity') from error


class AdoptedLogger:
    def __init__(self,row):
        self.process=psutil.Process(row['logger_pid']);self.returncode=None
        args=self.process.cmdline();track=str(Path(row['tracking_directory']).resolve())
        if '--output' not in args or str(Path(args[args.index('--output')+1]).resolve())!=track:
            raise ValueError('existing CPU logger PID command belongs to another tracking directory')
        if not any(Path(arg).name in ('track_relation_qa.py','track_preblock_online.py','track_preblock_online_r02.py','reattach_stream.py') for arg in args):
            raise ValueError('existing CPU logger PID is not an approved original observer')
        if row.get('logger_started_unix') is not None and abs(self.process.create_time()-row['logger_started_unix'])>5:
            raise ValueError('original CPU logger PID creation identity differs')
    def poll(self):
        if self.process.is_running():return None
        self.returncode=0;return self.returncode


def prepare_launch(row,c,root,campaign,expected_sha,repo,tracker,scope_override=None,*,host='fotile@121.196.228.153',port=12345):
    validate_tracker(tracker)
    if scope_override is not None and row['task']['id'] not in scope_override['active_quality_task_ids']:
        raise ValueError('excluded task may not start under user scope override')
    task=row['task'];ident=task['id'];directory=root/'controller-r02'/ident
    directory.mkdir(parents=True,exist_ok=False)
    run=c['control_root']+'/quality-r01/'+ident;snapshot=c['control_root']+'/efficiency-source-r01'
    ds=c['datasets'][task['encoder']][task['dataset']]
    # Original comparator checks actual artifacts/head/sampling before truth; this
    # additional stdlib check also precedes any quality process launch.
    quality=Path(c['output_root'])/ident;dense=Path(c['output_root'])/(task['encoder']+'-'+task['dataset']+'-dense')
    gate="import hashlib,json\nfrom pathlib import Path\n"
    gate+=f"roots=[Path({dense.as_posix()!r}),Path({quality.as_posix()!r})]\n"
    gate+="items=[]\nfor root in roots:\n if json.loads((root/'task.json').read_text())['status']!='prediction_completed':raise ValueError('original paired prediction incomplete')\n value=json.loads((root/'scoring/predictions/resolved.json').read_text());binding=value['artifacts']['predictions'];path=root/'scoring/predictions/predictions.jsonl'\n if Path(binding['path'])!=path or hashlib.sha256(path.read_bytes()).hexdigest()!=binding['sha256']:raise ValueError('actual prediction artifact differs')\n items.append(value)\n"
    gate+=f"if any(v['checkpoint_sha256']!={ds['head_checkpoint']['sha256']!r} for v in items):raise ValueError('actual paired head differs')\n"
    gate+="if items[0]['evaluation_sampling_fingerprint']!=items[1]['evaluation_sampling_fingerprint'] or items[0]['test_sequence']!=items[1]['test_sequence']:raise ValueError('actual paired input/head execution differs')\n"
    script=f'''set -euo pipefail
[ "$(hostname)" = ibnode3 ]
/users/fotile/VAD/.encoder-envs/v2/classic-video-v2/bin/python - <<'PAIR_GATE'
{gate}PAIR_GATE
RUN={shlex.quote(run)}
[ ! -e "$RUN" ];mkdir -p "$RUN"
printf '%s\\n' __SWANLAB_URL__ > "$RUN/swanlab-url.txt"
cat > "$RUN/run.sh" <<'RUNNER'
set +e
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
export PYTHONPATH={shlex.quote(snapshot+'/src')}
/users/fotile/VAD/.encoder-envs/v2/classic-video-v2/bin/python -u {shlex.quote(snapshot+'/scripts/icassp2027/run_pre_block_quality.py')} --campaign {shlex.quote(c['control_root']+'/campaign-r02.json')} --campaign-sha256 {expected_sha} --task {shlex.quote(ident)} --tracking-url "$VAD_TRACKING_URL"
rc=$?;echo "$rc" > {shlex.quote(run+'/exit-code.txt')};exit "$rc"
RUNNER
export VAD_TRACKING_URL=$(cat "$RUN/swanlab-url.txt")
nohup bash "$RUN/run.sh" > "$RUN/task.log" 2>&1 < /dev/null &
pid=$!;echo "$pid" > "$RUN/pid.txt"
echo "CPU_QUALITY_PID=$pid TASK={ident}"
tail --pid="$pid" -n +1 -f "$RUN/task.log"
exit "$(cat "$RUN/exit-code.txt")"
'''
    (directory/'launch.sh').write_text(script,encoding='utf-8')
    archive=campaign.parent/'efficiency-bundle-r01.tar.gz'
    provenance={'phase':'preblock_quality','remote_run':run,'task':task,'campaign_sha256':expected_sha,
                'expected_snapshot':snapshot,'formal_prediction_snapshot':c['snapshot_root'],'head':ds['head_checkpoint'],
                'truth':ds['truth'],'bootstrap_seed':0,'bootstrap_draws':10000,'head_training':False,
                'primary_metric':'frame_roc_auc' if task['dataset']=='ucf_crime' else 'frame_pr_auc',
                'step_AP_supplementary':True,'source_archive_sha256':sha(archive),
                'controller_source_sha256':sha(__file__),'tracker_source_sha256':sha(tracker),'short_poll_helper_sha256':sha(tracker_helper(tracker)),
                'transport':{'host':host,'port':port,'mode':'readonly_short_poll'},
                'formal_quality_pretruth_gate':'actual_head_sampling_test_sequence_and_prediction_SHA_then_original_frozen_compare_gate'}
    if scope_override is not None:provenance['user_scope_override']=scope_override
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    args=[sys.executable,str(tracker),'--launch-script',str(directory/'launch.sh'),'--provenance',str(directory/'provenance.json'),
          '--source-archive',str(archive),'--output',str(directory/'tracking'),'--name',ident+'-paired-quality',
          '--project',PROJECT,'--group','paired-quality-72','--completion-key','quality/completed','--failure-key','quality/failed',
          '--job-type','formal-cpu-paired-quality','--formal-quality','--host',host,'--port',str(port)]
    return directory,args,run,snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--sha',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--audit-only',action='store_true')
    p.add_argument('--scope-override',type=Path);p.add_argument('--scope-sha')
    p.add_argument('--host',default='fotile@121.196.228.153');p.add_argument('--port',type=int,default=12345)
    p.add_argument('--tracker',type=Path,default=Path(__file__).parent/'track_preblock_online_r02.py');a=p.parse_args()
    if sha(a.campaign)!=a.sha:raise ValueError('frozen campaign differs')
    c=read(a.campaign);root=a.root.resolve();repo=Path.cwd().resolve();cached=a.campaign.parent/'quality-results'
    if a.resume:state=read(root/'queue.json')
    else:
        if a.audit_only:raise ValueError('audit-only requires original resume state')
        root.mkdir(parents=True,exist_ok=False)
        state={'campaign_sha256':a.sha,'jobs':[{'task':t,'status':'pending'} for t in c['tasks'] if t['method']!='identity']}
    validate_rows(state,c,a.sha)
    scope=load_scope_override(a.scope_override,a.scope_sha,c,a.sha)
    active_ids=apply_scope_override(state,scope)
    probe=lambda row:ssh_probe(row,c,host=a.host,port=a.port)
    for row in state['jobs']:
        if row['status']=='completed':completion_evidence(row,c,cached,probe)
    if a.audit_only:
        print(json.dumps({'completed_reused':sum(r['status']=='completed' for r in state['jobs']),
                          'nonpending_uncertain':[r['task']['id'] for r in state['jobs'] if r['status'] not in ('completed','pending','cancelled_user_scope_change')],
                          **quality_counts(state,active_ids)}));return
    old_pid=state.get('controller_pid')
    if old_pid and psutil.pid_exists(old_pid):raise ValueError('old quality controller PID still exists; inspect before recovery')
    lock=root/'quality-controller-r02.lock'
    with lock.open('x',encoding='utf-8') as handle:handle.write(str(os.getpid()))
    state.update(controller_pid=os.getpid(),controller_recovered_from_pid=old_pid,controller_source_sha256=sha(__file__))
    child=None;active=None
    def save():
        state.update(quality_counts(state,active_ids));state['updated_unix']=time.time()
        atomic_json_write(root/'queue.json',state)
    try:
        uncertain=[r for r in state['jobs'] if r['status'] not in ('completed','pending','cancelled_user_scope_change')]
        if len(uncertain)>1:raise ValueError('multiple original uncertain CPU jobs; no additional CPU work permitted')
        if uncertain:
            active=uncertain[0]
            source_link(active)
            if active['expected_snapshot']!=c['control_root']+'/efficiency-source-r01':
                raise ValueError('uncertain original CPU snapshot differs; hold without replay')
            if active.get('logger_pid') and psutil.pid_exists(active['logger_pid']):child=AdoptedLogger(active)
            else:
                active['original_remote_probe']=probe(active)
                if active['original_remote_probe']['live']:
                    active['status']='held_inspect_original_cpu_pid';state['status']='held_inspect_original_cpu_pid';save();return
                try:completion_evidence(active,c,cached,probe)
                except Exception:
                    active['status']='held_inspect_original_cpu_pid';state['status']='held_inspect_original_cpu_pid';save();return
                active['status']='completed';active=None
        while True:
            if child is not None:
                track=Path(active['tracking_directory'])
                if (track/'swanlab.json').exists():active['swanlab']=read(track/'swanlab.json')
                if child.poll() is not None:
                    try:
                        completion_evidence(active,c,cached,probe)
                        if child.returncode!=0:raise ValueError('original CPU logger returned nonzero')
                    except Exception as exc:
                        active['status']='held_inspect_original_cpu_pid';active['observer_error']=str(exc)
                        state['status']='held_inspect_original_cpu_pid';save();return
                    active.update(status='completed',logger_exit=child.returncode,cloud_verification=read(track/'cloud-verification.json'))
                    child=None;active=None
            if child is None:
                active=next_ready(state,read(a.campaign.parent/'queue.json'),active_ids)
                if active is not None:
                    check=ssh_probe(active,c,pending=True,host=a.host,port=a.port)
                    if check['remote_run_exists'] or check['quality_exists']:
                        active['original_remote_probe']=check;active['status']='held_inspect_original_cpu_pid';state['status']='held_inspect_original_cpu_pid';save();return
                    directory,args,run,snapshot=prepare_launch(active,c,root,a.campaign,a.sha,repo,a.tracker.resolve(),scope,host=a.host,port=a.port)
                    with (directory/'logger.stdout').open('w',encoding='utf-8') as out,(directory/'logger.stderr').open('w',encoding='utf-8') as err:
                        child=subprocess.Popen(args,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    (directory/'logger.pid').write_text(str(child.pid))
                    active.update(status='running',logger_pid=child.pid,logger_started_unix=time.time(),tracking_directory=str(directory/'tracking'),
                                  swanlab_source=str(directory/'tracking/swanlab.json'),expected_snapshot=snapshot,remote_run=run)
                state['status']='running' if child is not None else 'waiting_for_matched_predictions'
            save()
            if state['active_completed']==state['active_quality_total'] and child is None:state['status']='completed';save();return
            time.sleep(30)
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()


if __name__=='__main__':main()

"""Resume original online SwanLab identity from read-only short SSH log polls.

Never starts/stops a remote job. Missing/changed PID identity, a lost log prefix,
or ambiguous terminal state is a hold, not permission to replay GPU work.
"""
import argparse, base64, hashlib, json, os, shlex, subprocess, time
from pathlib import Path


def canonical(event):
    return json.dumps(event, sort_keys=True, separators=(',', ':'))


def read_events(path, seen=None):
    path=Path(path).resolve()
    seen=set() if seen is None else seen
    if path in seen:raise ValueError('cyclic original tracking recovery chain')
    seen.add(path)
    prefix=[]
    recovery=path.parent/'recovery.json'
    if recovery.exists():
        old=json.loads(recovery.read_text(encoding='utf-8'))
        prior=Path(old['previous_tracking'])
        current_link=json.loads((path.parent/'swanlab.json').read_text(encoding='utf-8'))
        prior_link=json.loads((prior/'swanlab.json').read_text(encoding='utf-8'))
        if current_link!=prior_link or old['original_swanlab']!=current_link:
            raise ValueError('original SwanLab identity differs within recovery chain')
        prefix=read_events(prior/'events.jsonl',seen)
    return prefix+[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


class EventCursor:
    def __init__(self, previous):
        self.previous = previous
        self.matched = 0
        self.pending = b''

    def feed(self, raw):
        self.pending += raw
        lines = self.pending.split(b'\n')
        self.pending = lines.pop()
        fresh = []
        for raw_line in lines:
            line = raw_line.decode('utf-8', errors='replace')
            if not line.startswith('EVENT '):
                continue
            event = json.loads(line[6:])
            if self.matched < len(self.previous):
                if canonical(event) != canonical(self.previous[self.matched]):
                    raise ValueError('original remote EVENT prefix differs; hold without replay')
                self.matched += 1
            else:
                fresh.append(event)
        return fresh


def remote_script(args, offset, identity):
    settings = dict(log=args.remote_log, pid_file=args.remote_pid_file,
                    marker=args.expected_command_marker, owner=args.owner,
                    offset=offset, identity=identity, expected_url=args.expected_url)
    code = r'''
import base64,json,os,pwd
from pathlib import Path
s=SETTINGS
log=Path(s['log']); pidfile=Path(s['pid_file']); root=pidfile.parent
if os.uname().nodename!='ibnode3':raise ValueError('wrong node')
identity_root=root.parent if root.name=='data' else root
url=(identity_root/'swanlab-url.txt').read_text().strip()
if url!=s['expected_url']:raise ValueError('original remote SwanLab URL differs')
pid=int(pidfile.read_text().strip());proc=Path('/proc')/str(pid)
live=proc.exists();identity=None;command=None
if live:
    if pwd.getpwuid(proc.stat().st_uid).pw_name!=s['owner']:raise ValueError('original PID owner differs')
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
    if s['marker'] not in command:raise ValueError('original PID command marker differs')
    # Field22 follows the last close parenthesis in Linux /proc/PID/stat.
    identity={'pid':pid,'start_ticks':(proc/'stat').read_text().rsplit(')',1)[1].split()[19]}
    if s['identity'] is not None and identity!=s['identity']:raise ValueError('original PID was reused')
st=log.stat()
if s['offset']>st.st_size:raise ValueError('original log was truncated')
with log.open('rb') as f:
    f.seek(s['offset']);raw=f.read(4*1024*1024)
exitpath=identity_root/'exit-code.txt';exitcode=int(exitpath.read_text().strip()) if exitpath.exists() else None
queue=(root/'queue.json') if root.name=='data' else (root/'data/queue.json');queue_terminal=False
if queue.exists():
    q=json.loads(queue.read_text())
    queue_terminal=(q.get('status')=='completed' and len(q.get('jobs',[]))==456
      and all(e.get('status')=='completed' for e in q['jobs']))
print(json.dumps({'live':live,'identity':identity,'command':command,'owner':s['owner'],
 'remote_url':url,'log_identity':[st.st_dev,st.st_ino],'log_size':st.st_size,
 'chunk':base64.b64encode(raw).decode(),'exit_code':exitcode,'queue_terminal':queue_terminal}))
'''.replace('SETTINGS', repr(settings))
    return 'python3 - <<\'PY\'\n' + code + '\nPY\n'


def poll(args, offset, identity):
    command = ['ssh', '-T', '-p', str(args.port), '-o', 'BatchMode=yes', '-o',
               'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', args.host, 'bash', '-s']
    for attempt in range(3):
        try:
            result = subprocess.run(command, input=remote_script(args, offset, identity).encode(),
                                    capture_output=True, timeout=35)
        except subprocess.TimeoutExpired as exc:
            error = exc
        else:
            if result.returncode == 0:
                return json.loads(result.stdout)
            text = result.stderr.decode(errors='replace')
            if any(word in text.lower() for word in ('permission denied', 'host key verification failed',
                      'remote host identification has changed', 'no such identity')):
                raise RuntimeError('SSH authentication/host verification failed; no retry: ' + text)
            if result.returncode != 255:
                raise RuntimeError('read-only remote identity/log check failed: ' + text)
            error = RuntimeError('SSH transport failure: ' + text)
        if attempt < 2:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError('three short-poll transport attempts exhausted; inspect same original PID') from error


def verify_cloud(swanlab, link, output, completion_key, succeeded, error):
    result = dict(link, verified=False, error=error)
    api = swanlab.Api()
    for attempt in range(3):
        remote = api.run(link['path'])
        summary = remote.summary(keys=[completion_key])
        result.update(state=remote.state, completion_summary=summary, attempts=attempt+1)
        if succeeded and remote.state.upper() == 'FINISHED' and summary.get(completion_key, {}).get('value') == 1:
            result['verified'] = True
            break
        if attempt < 2:
            time.sleep(5)
    (output/'cloud-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('CLOUD_VERIFICATION ' + json.dumps(result), flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('previous-tracking','swanlab-source','output','provenance'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('remote-log','remote-pid-file','expected-command-marker','completion-key','failure-key'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--project',default='VAD-PreBlock-VMA-TS-20261015')
    p.add_argument('--host',default='fotile@121.196.228.153');p.add_argument('--port',type=int,default=12345)
    p.add_argument('--owner',default='fotile');p.add_argument('--poll-seconds',type=float,default=30)
    a=p.parse_args()
    source=a.swanlab_source/'swanlab.json' if a.swanlab_source.is_dir() else a.swanlab_source
    link=json.loads(source.read_text(encoding='utf-8'))
    parts=link['path'].strip('/').split('/')
    if len(parts)!=3 or parts[1]!=a.project or parts[2]!=link['run_id']:
        raise ValueError('original SwanLab identity/project differs')
    if link['url'].rstrip('/') != f'https://swanlab.cn/@{parts[0]}/{parts[1]}/runs/{parts[2]}':
        raise ValueError('original SwanLab URL/path/runid inconsistent')
    original=json.loads((a.previous_tracking/'swanlab.json').read_text(encoding='utf-8'))
    if original!=link:raise ValueError('previous tracking/source refer to different original runs')
    previous=read_events(a.previous_tracking/'events.jsonl')
    provenance=json.loads(a.provenance.read_text(encoding='utf-8'))
    a.expected_url=link['url']
    # Validate exact live process / remote URL before altering any cloud state.
    first=poll(a,0,None)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'swanlab.json').write_text(json.dumps(link,indent=2),encoding='utf-8')
    recovery={'previous_tracking':str(a.previous_tracking.resolve()),'original_swanlab':link,
              'original_pid_file':a.remote_pid_file,'original_log':a.remote_log,
              'previous_events':len(previous),'provenance':provenance,'pid_identity':first['identity'],
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tracking_mode':'online_resume_must_short_readonly_ssh_no_job_replay'}
    (a.output/'recovery.json').write_text(json.dumps(recovery,indent=2),encoding='utf-8')
    import swanlab
    from swanlab.sdk.internal.settings import Settings
    run=swanlab.init(project=a.project,workspace=parts[0],id=link['run_id'],resume='must',
       mode='online',public=False,log_dir=str(a.output/'swanlog'),
       settings=Settings(interactive=False,probe=Settings.Probe(monitor=False,hardware=False)))
    if run.id!=link['run_id'] or run.path.strip('/')!=link['path'].strip('/') or run.url!=link['url']:
        run.finish(state='crashed',error='resume identity mismatch')
        raise ValueError('online resume returned another identity')
    run.config.update({'transport_recovery':recovery})
    for artifact in (Path(__file__),a.provenance,a.output/'recovery.json'):
        run.save(artifact.resolve(),base_path=artifact.resolve().parent,policy='now')
    cursor=EventCursor(previous);offset=0;identity=first['identity'];log_identity=first['log_identity']
    # Old logger wrote its transport error at len(events)+1. Start strictly later.
    old_remote=swanlab.Api().run(link['path'])
    old_summary=old_remote.summary(keys=[a.failure_key,a.completion_key])
    old_steps=[int(value.get('step',0)) for value in old_summary.values() if isinstance(value,dict)]
    step=max([len(previous)+2,*[n+1 for n in old_steps]])
    run.log({a.failure_key:0,'tracking/transport_failed':0,'tracking/reattached_original_pid':1},step=step)
    run.log_text(key='audit/transport_recovery',data=json.dumps(recovery),step=step)
    completed=any(e.get('metrics',{}).get(a.completion_key)==1 for e in previous)
    failed=False;error=None;succeeded=False
    try:
        with (a.output/'remote.log').open('wb') as rawlog,(a.output/'events.jsonl').open('w',encoding='utf-8') as events:
            sample=first
            while True:
                if sample['log_identity']!=log_identity:raise ValueError('original log inode changed')
                if identity is None and sample['identity'] is not None:raise ValueError('PID appeared after original terminal observation')
                chunk=base64.b64decode(sample['chunk']);rawlog.write(chunk);rawlog.flush();offset+=len(chunk)
                for event in cursor.feed(chunk):
                    step+=1
                    if event.get('metrics'):run.log(event['metrics'],step=step)
                    run.log_text(key='audit/events',data=json.dumps(event,ensure_ascii=False),step=step)
                    events.write(json.dumps(event,ensure_ascii=False)+'\n');events.flush()
                    completed=completed or event.get('metrics',{}).get(a.completion_key)==1
                    failed=failed or event.get('metrics',{}).get(a.failure_key)==1
                    print('EVENT '+json.dumps(event,ensure_ascii=False),flush=True)
                state={'offset':offset,'remote_log_identity':log_identity,'pid_identity':identity,
                       'prefix_events_matched':cursor.matched,'previous_events':len(previous),
                       'step':step,'remote_live':sample['live'],'completed_event_seen':completed,
                       'exit_code':sample['exit_code'],'queue_terminal':sample['queue_terminal']}
                (a.output/'checkpoint.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
                drained=offset==sample['log_size']
                if drained and cursor.matched!=len(previous):raise ValueError('original uploaded event prefix absent from remote log')
                if drained and cursor.pending and not sample['live']:raise ValueError('terminal original log has incomplete final line')
                terminal_receipt=sample['exit_code']==0 or sample['queue_terminal']
                if drained and completed and terminal_receipt and not failed:
                    succeeded=True;break
                if drained and not sample['live']:
                    raise RuntimeError('original PID absent without matching completion plus exit/queue receipt; hold')
                if drained and sample['exit_code'] is not None and sample['exit_code']!=0:
                    raise RuntimeError('original job exited nonzero; hold without retry')
                if drained:time.sleep(a.poll_seconds)
                sample=poll(a,offset,identity)
        for artifact in ('remote.log','events.jsonl','recovery.json','checkpoint.json'):
            run.save((a.output/artifact).resolve(),base_path=a.output.resolve(),policy='now')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        run.log({'tracking/transport_failed':1},step=step+1)
    finally:
        run.finish(state='success' if succeeded else 'crashed',error=error)
    verification=verify_cloud(swanlab,link,a.output,a.completion_key,succeeded,error)
    if error or not verification['verified']:
        raise RuntimeError(error or 'original cloud terminal verification pending; hold without GPU replay')


if __name__=='__main__':main()

"""First GPU execution only, after transport failed before remote creation; preserve original Swan identity."""
import argparse,hashlib,json,runpy,subprocess,sys
from pathlib import Path

def offset_logger(log,base):
 def shifted(data,step=None,**kwargs):
  return log(data,step=None if step is None else step+base,**kwargs)
 return shifted

def main():
 p=argparse.ArgumentParser();p.add_argument('--original-swanlab',type=Path,required=True);p.add_argument('--frozen-tracker',type=Path,required=True);a,rest=p.parse_known_args()
 def arg(k):return rest[rest.index(k)+1]
 link=json.loads(a.original_swanlab.read_text());provenance=json.loads(Path(arg('--provenance')).read_text());task=provenance['task'];campaign=provenance['campaign']
 manifest_path=Path(__file__).with_name('prelaunch-resume-manifest-r02.json')
 if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!='06490a8176dca70dfcb33a2a0792079a0033f08048db1bce3ad518efd5ee5a23':raise ValueError('approved first-execution identity manifest changed')
 approved=json.loads(manifest_path.read_text())['tasks'].get(task['id'])
 if approved is None or approved['task']!=task or approved['swanlab']!=link or approved['remote_run']!=provenance['remote_run']:raise ValueError('only exact proven unstarted original identity may resume')
 if hashlib.sha256(a.frozen_tracker.read_bytes()).hexdigest()!='3cfc843047752b00f71c85389b951296f1970823bc2e1d56dac733bfb29fdb99':raise ValueError('frozen observer changed')
 remote=provenance['remote_run'];output=campaign['output_root']+'/'+task['id']
 code="import os;from pathlib import Path;assert os.uname().nodename=='ibnode3';assert not Path("+repr(remote)+").exists();assert not Path("+repr(output)+").exists();print('ORIGINAL_RUN_AND_OUTPUT_ABSENT')"
 command=['ssh','-T','-p',arg('--port'),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',arg('--host'),'python3','-']
 result=subprocess.run(command,input=code.encode(),capture_output=True,timeout=35)
 if result.returncode!=0 or result.stdout.strip()!=b'ORIGINAL_RUN_AND_OUTPUT_ABSENT':raise RuntimeError('original remote absence not proven; hold without GPU launch')
 import swanlab
 api=swanlab.Api();old=api.run(link['path'])
 if old.name!=link['name'] or not old.run_id:raise ValueError('original cloud run not verified')
 summary=old.summary();base=max([0]+[int(v.get('step',0)) for v in summary.values() if isinstance(v,dict)])+1
 response=old._ctx.client.post('/project/'+link['path'].rsplit('/',1)[0]+'/experiment',{'name':old.name,'cuid':old.run_id},retries=0)
 if response.raw.status_code!=200 or response.data['cuid']!=old.run_id or response.data.get('slug')!=link['run_id']:raise ValueError('must resume same existing original run')
 original_init=swanlab.init
 def init_same(**kwargs):
  kwargs.update(id=link['run_id'],resume='must',workspace=link['path'].split('/')[0],name=link['name'])
  run=original_init(**kwargs)
  if run.id!=link['run_id'] or run.path.strip('/')!=link['path'] or run.url!=link['url']:raise ValueError('original cloud identity changed; no GPU')
  run.save(Path(__file__).resolve(),base_path=Path(__file__).resolve().parent,policy='now')
  run.save(manifest_path.resolve(),base_path=manifest_path.resolve().parent,policy='now')
  run.log({'test/failed':0,'tracking/first_execution_after_verified_absence':1},step=base)
  run.log=offset_logger(run.log,base)
  original_text=run.log_text
  def text_shift(*args,step=None,**kwargs):return original_text(*args,step=None if step is None else step+base,**kwargs)
  run.log_text=text_shift
  return run
 swanlab.init=init_same
 sys.argv=[str(a.frozen_tracker),*rest]
 runpy.run_path(str(a.frozen_tracker),run_name='__main__')
if __name__=='__main__':main()

import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
for i,c in enumerate(json.loads((r/'commands.json').read_text())):
 with (r/f'gate-{i}.stdout').open('w',encoding='utf-8') as out,(r/f'gate-{i}.stderr').open('w',encoding='utf-8') as err:
  p=subprocess.Popen(c,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
  (r/f'gate-{i}.pid').write_text(str(p.pid));rc=p.wait()
 if rc:raise RuntimeError(f'gate {i} failed; inspect original PID, no retry')
(r/'completed.json').write_text(json.dumps({'all3_gates_completed':True}))

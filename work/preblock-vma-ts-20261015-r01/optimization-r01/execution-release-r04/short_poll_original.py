"""Read an already-launched formal job; never execute/replay a remote job."""
import base64
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / 'transport-recovery-r01'))
from reattach_stream import poll


def stream_original_task(remote_run, url, host, port, poll_seconds=30):
    args = SimpleNamespace(remote_log=remote_run + '/task.log',
                           remote_pid_file=remote_run + '/pid.txt',
                           expected_command_marker=remote_run + '/run.sh',
                           expected_url=url, owner='fotile', host=host, port=port)
    offset = 0
    identity = None
    log_identity = None
    pending = b''
    completed = False
    while True:
        state = poll(args, offset, identity)
        if log_identity is None:
            log_identity = state['log_identity']
            identity = state['identity']
        elif state['log_identity'] != log_identity:
            raise ValueError('original remote log inode changed; no replay')
        chunk = base64.b64decode(state['chunk'])
        offset += len(chunk)
        pending += chunk
        lines = pending.split(b'\n')
        pending = lines.pop()
        for raw_line in lines:
            line = raw_line.decode('utf-8', errors='replace')
            if line.startswith('EVENT '):
                event = json.loads(line[6:])
                completed = completed or event.get('metrics', {}).get('test/completed') == 1
            yield line + '\n'
        drained = offset == state['log_size']
        if drained and completed and state['exit_code'] == 0 and not pending:
            return
        if drained and (not state['live'] or state['exit_code'] not in (None, 0)):
            raise RuntimeError('original job terminal without complete receipt; inspect, never replay')
        if drained:
            time.sleep(poll_seconds)

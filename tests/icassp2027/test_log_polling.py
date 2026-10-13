import json
from types import SimpleNamespace
import pytest
from scripts.icassp2027 import attach_relation_run as recovery


def event(kind, value):
    return 'EVENT '+json.dumps({'kind':kind,'metrics':{'v':value}})+'\n'


def test_short_connections_resume_cursor_without_gpu_replay(monkeypatch):
    replies=iter([event('progress',1)+event('progress',2)+'REMOTE_LOG_STATE 2 1\n',
                  event('completed',3)+'REMOTE_LOG_STATE 3 0\n'])
    scripts=[]
    def run(cmd,**kwargs):
        scripts.append(kwargs['input'].decode())
        return SimpleNamespace(returncode=0,stdout=next(replies).encode())
    monkeypatch.setattr(recovery.subprocess,'run',run)
    monkeypatch.setattr(recovery.time,'sleep',lambda _:None)
    last={'kind':'progress','metrics':{'v':1}}
    lines=list(recovery.poll_original_log('/base','task','/snapshot',last))
    assert lines==[event('progress',2),event('completed',3)]
    assert 'tail -n +3' in scripts[1]
    assert all('nohup' not in s and 'kill ' not in s for s in scripts)


def test_completed_cursor_reacknowledges_cloud_only(monkeypatch):
    text=event('completed',1)+'REMOTE_LOG_STATE 1 0\n'
    monkeypatch.setattr(recovery.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=text.encode()))
    lines=list(recovery.poll_original_log('/base','task','/snapshot',{'kind':'completed','metrics':{'v':1}}))
    assert lines==[event('completed',1)]


@pytest.mark.parametrize('message',['Permission denied','Host key verification failed'])
def test_authentication_failure_never_retried(monkeypatch,message):
    calls=[]
    def run(*a,**k):calls.append(1);return SimpleNamespace(returncode=255,stdout=message.encode())
    monkeypatch.setattr(recovery.subprocess,'run',run)
    with pytest.raises(RuntimeError):list(recovery.poll_original_log('/base','task','/snapshot'))
    assert len(calls)==1

import json
import psutil
import pytest
from scripts.icassp2027.schedule_relation_tests import ExistingTracker


class Process:
    running=True
    def create_time(self):return 123
    def is_running(self):return self.running


def test_adopted_online_logger_is_not_restarted(monkeypatch,tmp_path):
    process=Process();monkeypatch.setattr(psutil,'Process',lambda pid:process)
    tracker=ExistingTracker(5,tmp_path)
    assert tracker.poll() is None
    process.running=False
    assert tracker.poll()==1
    (tmp_path/'cloud-verification.json').write_text(json.dumps({'verified':True,'state':'FINISHED'}))
    assert tracker.poll()==0


@pytest.mark.parametrize('state,verified',[('RUNNING',True),('CRASHED',True),('FINISHED',False)])
def test_adopted_logger_without_cloud_success_holds(monkeypatch,tmp_path,state,verified):
    process=Process();process.running=False;monkeypatch.setattr(psutil,'Process',lambda pid:process)
    tracker=ExistingTracker(5,tmp_path)
    (tmp_path/'cloud-verification.json').write_text(json.dumps({'verified':verified,'state':state}))
    assert tracker.poll()==1

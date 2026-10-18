"""Bounded retry for the observed Windows reader/antivirus rename lock."""
import json,os,time


def atomic_json_write(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    for attempt in range(10):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==9:raise
            time.sleep(min(.1*2**attempt,1.))

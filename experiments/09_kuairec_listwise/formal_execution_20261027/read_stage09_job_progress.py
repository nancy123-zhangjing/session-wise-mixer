"""Read the existing Stage09 Job through the official SDK; no remote writes."""
import contextlib, datetime, hashlib, io, json, logging, shlex
from pathlib import Path
from inspire import InspireClient

command = '''from pathlib import Path
from collections import deque
from datetime import datetime, timezone
import json
g=Path("/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/goal_20261027_r1")
out={"observed_at_utc":datetime.now(timezone.utc).isoformat(), "arms":{}}
for a in ("J0","J1","H0","H1"):
    with (g/"runs"/a/"training.jsonl").open() as f: training=json.loads(deque(f,maxlen=1)[0])
    with (g/"delivery/four_arm_supervisor_r1"/("train_"+a+".log")).open() as f: latest=deque(f,maxlen=1)[0].strip()
    p=g/"runs"/a/"validation_R08/summary.json"
    out["arms"][a]={"training":training,"last_log":latest,"R08_summary":json.loads(p.read_text()) if p.is_file() else None,"run_complete":(g/"runs"/a/"complete.json").is_file()}
print("STAGE09_PROGRESS_JSON="+json.dumps(out,separators=(",",":")))
'''
logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    with InspireClient(account="sii-shared",allow_browser=False,timeout=15,operation_timeout=60) as c:
        job=c.jobs.get("nancy-stage09-four-arm-20261027-r1", workspace="分布式训练空间")
        assert job.status.upper()=="RUNNING", "Existing Job is not RUNNING"
        result=c.jobs.exec(job.ref,command="python -B -c "+shlex.quote(command),timeout=40,max_output_bytes=65536)
assert result.completed and result.returncode==0 and not result.truncated
lines=[line[len("STAGE09_PROGRESS_JSON="):] for line in result.stdout.splitlines() if line.startswith("STAGE09_PROGRESS_JSON=")]
assert len(lines)==1
data=json.loads(lines[0])
data.update(job_name=job.name,job_status=job.status,channel="Official SDK Job exec; read-only")
stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
p=Path(__file__).with_name("FORMAL_PROGRESS_"+stamp+".json")
with p.open("x",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2);f.write("\n")
print(json.dumps({"file":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),**data},ensure_ascii=False))

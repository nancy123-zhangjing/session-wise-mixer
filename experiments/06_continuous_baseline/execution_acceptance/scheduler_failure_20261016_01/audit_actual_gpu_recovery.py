from pathlib import Path
import csv,hashlib,io,json,tarfile,time
base=Path("/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage06/goal_20261016_precision_v2")
pipeline=base/"code/outputs/stage06_continuous/pipeline_20261016_precision_v2_supervisor_v3"
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gate=read(pipeline/"smoke/GPU_GATE.json")
assert gate["status"]=="PASS" and gate["source_sha256"]=="b240275a547d36537282f318099241abf9b0be9a197489b040f8b41b6ae6c907"
inventory=gate["inventory"]
assert len(inventory)==len({r["uuid"] for r in inventory})==8
assert all("H100" in r["name"] for r in inventory)
assert len(gate["gradient_checks"])==4 and all(r["relative_gradient_l2"]<.05 and r["gradient_cosine"]>.999 for r in gate["gradient_checks"])
stress=read(pipeline/"smoke/NUMERICAL_STRESS.json")
assert stress==gate["numerical_stress"] and stress["status"]=="PASS" and len(stress["epochs"])==35
assert stress["attention_precision_policy"]=="fp32_math" and not stress["validation_or_test_used"]
assert all(r["updates"]==514 and r["samples"]==131413 and r["maximum_gradient_norm"]<1e6 for r in stress["epochs"])
prepared=read(pipeline/"RECOVERY_PREPARED.json")
q=read(pipeline/"campaign/queue.json")
assert q["status"]=="RUNNING" and q["source_sha256"]==gate["source_sha256"]
deployment=base/"orchestration_v3/DEPLOYMENT_MANIFEST.json"
assert q["deployment_manifest_sha256"]==sha(deployment)
assert q["supervisor_sha256"]==read(deployment)["supervisor_sha256"]==sha(base/"orchestration_v3/scripts/train/continuous_queue_supervisor.py")
entries={}
for rel in ["smoke/GPU_GATE.json","smoke/NUMERICAL_STRESS.json","PIPELINE.json","RECOVERY_PREPARED.json","campaign/queue.json"]:
 entries[rel]=(pipeline/rel).read_bytes()
records=[]
for r in prepared["resumed_tasks"]:
 task=next(t for t in q["tasks"] if t["task_id"]==r["task_id"])
 output=Path(r["destination"]); old=Path(r["source"])
 events=[json.loads(line) for line in (output/"resume_events.jsonl").read_text().splitlines()]
 event=events[-1]
 assert event["gpu_uuid"]==event["cuda_visible_devices"]==q["gpu_inventory"][task["gpu"]]["uuid"]
 assert event["pid"]>0 and event["device"]=="cuda:0"
 before=list(csv.DictReader((old/"curves.csv").open()))
 after=list(csv.DictReader((output/"curves.csv").open()))
 assert len(before)==r["checks"]["last"]["epoch"] and len(after)>len(before) and after[:len(before)]==before
 current=read(output/"CURRENT.json")
 assert current["binding"]["source_sha256"]==q["source_sha256"] and current["epoch"]>len(before)
 for name in ["resume_events.jsonl","run.json","CURRENT.json","curves.csv"]:
  entries[(output/name).relative_to(pipeline).as_posix()]=(output/name).read_bytes()
 records.append({"task_id":task["task_id"],"from_epoch":len(before),"observed_epoch":current["epoch"],"resume_event":event,"old_curve_prefix_unchanged":True})
proof={"status":"PASS","observed_unix":time.time(),"training_source_sha256":q["source_sha256"],"supervisor_sha256":q["supervisor_sha256"],"gpu_gate_sha256":sha(pipeline/"smoke/GPU_GATE.json"),"resumed_tasks":records,"retained_completed":38,"test_unlocked":False,"scope":"Real GPU gate receipts and eight actual resumed curve continuations; optimizer/RNG CPU states verified in preparation, CUDA RNG bytes not individually replayed by this audit"}
assert not (pipeline/"campaign/FROZEN_SELECTION.json").exists()
entries["GPU_RECOVERY_AUDIT.json"]=(json.dumps(proof,indent=2)+"\n").encode()
archive=base/"delivery/supervisor_v3_gpu_recovery_01.tar.gz"; manifest=archive.with_name("supervisor_v3_gpu_recovery_01.json")
assert not archive.exists() and not manifest.exists()
with tarfile.open(archive,"w:gz") as tf:
 for relative,data in entries.items():
  member=tarfile.TarInfo(relative);member.size=len(data);tf.addfile(member,io.BytesIO(data))
m={"status":"RECOVERY_EVIDENCE_ARCHIVED","archive_sha256":sha(archive),"files":{r:{"bytes":len(d),"sha256":hashlib.sha256(d).hexdigest()} for r,d in entries.items()}}
manifest.write_text(json.dumps(m,indent=2)+"\n")
print(json.dumps({"status":"PASS","retained_completed":38,"actual_resumed_tasks":len(records),"archive_sha256":m["archive_sha256"],"files":len(entries),"resumed":[{"task":r["task_id"],"before":r["from_epoch"],"after":r["observed_epoch"]} for r in records]}))

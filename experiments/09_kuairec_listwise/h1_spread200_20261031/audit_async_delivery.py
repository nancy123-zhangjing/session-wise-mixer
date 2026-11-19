"""Final CPU audit of the operational continuation from a verified fresh extraction.

Checks provenance and readonly task completeness; does not replace the full
scientific trajectory, metrics, cold restoration or resource closure audits.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extracted',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.extracted.resolve();run=root/'runs/H1-Spread200-R35'
    if read(root/'LOCAL_EXTRACTION_VERIFIED.json')['status']!='passed':raise ValueError('Fresh extraction not verified')
    package=root/'ASYNC_DEPLOYMENT.json'
    if sha(package)!='b9f829d07aded961e8508c129fb5af318e71c9bb78a2fc0c5d5ab626ac8b8b09':raise ValueError('Different operational package')
    for name,digest in read(package)['files'].items():
        path=root/name;path.resolve().relative_to(root)
        if sha(path)!=digest:raise ValueError('Operational source/evidence changed: '+name)
    gate=read(root/'ASYNC_RESUME_GATE.json')
    for name,digest in gate['evidence'].items():
        if sha(root/name)!=digest:raise ValueError('Recovery evidence changed')
    jobs=root/'jobs/formal_async_r2';result=read(jobs/'result.json')
    pointer=read(root/'FORMAL_COMPLETION.json')
    if (pointer['attempt']!='formal_async_r2' or pointer['result_sha256']!=sha(jobs/'result.json')
            or pointer['resume_gate_sha256']!=sha(jobs/'resume_gate.json')):raise ValueError('Completion pointer mismatch')
    saved_gate=read(jobs/'resume_gate.json')
    if (saved_gate['gate_sha256']!=sha(root/'ASYNC_RESUME_GATE.json') or
            saved_gate['launcher_sha256']!=sha(root/'job_async_r2.py')):raise ValueError('Launcher gate mismatch')
    if result['status']!='training_validation_test_predictions_complete' or result['stages']!=[
            dict(stage='train',exit_code=0),dict(stage='test',exit_code=0)]:raise ValueError('Formal continuation incomplete')
    if (result['runtime_sha256']!=sha(root/'async_resume.py') or result['launcher_sha256']!=sha(root/'job_async_r2.py')
            or result['automatic_retry'] is not False or result['stop_reason'] is not None):raise ValueError('Runtime outcome mismatch')
    expected={str(phase):(kind,epoch) for phase,kind,epoch in read(root/'CPU_ASYNC_STRUCTURE.json')['remaining_tasks']}
    requests=sorted(p for p in (jobs/'tasks').glob('phase*.json') if not p.name.endswith('.done.json'))
    if {str(int(path.stem[5:])) for path in requests}!=set(expected):raise ValueError('Readonly task coverage mismatch')
    verified=[]
    for path in requests:
        task,done=read(path),read(path.with_suffix('.done.json'))
        phase=str(int(path.stem[5:]))
        if (task['kind'],task['epoch'])!=expected[phase]:raise ValueError('Task kind/epoch mismatch')
        checkpoint=run/Path(task['checkpoint']).name
        receipt=read(run/'checkpoint_receipts'/(checkpoint.name+'.json'))
        if sha(checkpoint)!=task['checkpoint_sha256'] or receipt['sha256']!=sha(checkpoint):raise ValueError('Task checkpoint mismatch')
        if (done['status']!='completed_readonly_task' or done['request_sha256']!=sha(path)
                or done['checkpoint_sha256']!=sha(checkpoint) or done['runtime_sha256']!=gate['runtime_sha256']
                or done['state_before']!=done['state_after'] or done['state_before']!=receipt['state_sha256']):
            raise ValueError('Readonly task changed or loaded a different state')
        diagnostic=read(run/'diagnostics'/task['label']/'complete.json')
        if task['kind']=='validation':
            record=read(run/'validation_records'/f'R{task["epoch"]:02d}.json')
            if (record['checkpoint']!=task['checkpoint'] or record['checkpoint_sha256']!=sha(checkpoint)
                    or diagnostic['source_sha256']!=record['predictions_sha256'] or diagnostic['reused']!=record['predictions']):
                raise ValueError('Validation/diagnostic reuse mismatch')
        else:
            prediction=run/Path(diagnostic['predictions']).relative_to(Path(task['checkpoint']).parent)
            if sha(prediction)!=diagnostic['predictions_sha256']:raise ValueError('Diagnostic prediction changed')
        verified.append(dict(phase=int(phase),kind=task['kind'],epoch=task['epoch'],label=task['label'],
            checkpoint_sha256=task['checkpoint_sha256'],readonly_state_exact=True,receipt_sha256=sha(path.with_suffix('.done.json'))))
    first=read(root/'recovery_async_r2/FIRST30_ACCEPTED.json')
    if (first['status']!='passed_first30_real_formal_continuation' or not first['exact_loss_match_to_isolated_CUDA_oracle']
            or sha(root/'recovery_async_r2/first30_training.jsonl')!=first['training_snapshot_sha256']):raise ValueError('Initial continuation audit mismatch')
    report=dict(status='passed_async_operational_delivery_audit',runtime_sha256=gate['runtime_sha256'],
        package_sha256=sha(package),readonly_tasks=verified,first30_receipt_sha256=sha(root/'recovery_async_r2/FIRST30_ACCEPTED.json'),
        limitation='Provenance and readonly execution only; scientific metrics, all updates, fresh cold restoration and actual GPU release remain separately required.')
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()

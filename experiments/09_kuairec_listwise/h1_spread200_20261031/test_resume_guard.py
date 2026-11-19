"""CPU-only operational gate tests. All approvals below are temporary fixtures."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys
from resume_guard import validate_resume,PARENT_SHA
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v),encoding='utf-8');return sha(p)

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=HERE,prefix='resume_guard_fixture_')
        self.root=Path(self.tmp.name).resolve();self.root.relative_to(HERE)
        self.addCleanup(self.cleanup)
        self.recovery=self.root/'recovery_r2';self.recovery.mkdir()
        self.run=self.root/'runs/H1-Spread200-R35';self.run.mkdir(parents=True)
        self.cp=self.recovery/'resume.pt';self.cp.write_bytes(b'UNIT TEST ONLY checkpoint')
        (self.run/'training.jsonl').write_bytes(b'UNIT TEST ONLY journal')
        self.proof=dict(status='passed_CPU_restore_and_journal_alignment',checkpoint_sha256=sha(self.cp),journal_exactly_matches_checkpoint=True,replayed_rows_required=0,restored_equal={'model':True},r_updates=16339,t_updates=140,progress=dict(phase=30,update=400))
        self.terminal=dict(name='nancy-s09-d-1031-r1',status='CANCELLED')
        self.proof_path=self.recovery/'RESTORE_READY.json';self.terminal_path=self.recovery/'PLATFORM_STATUS.json'
        evidence=self.recovery/'fixture_evidence.txt';evidence.write_text('UNIT TEST FIXTURE; not a real administrator approval.',encoding='utf-8')
        self.approval=dict(status='approved',authority='workspace_administrator',scope='stage09_h1_spread200_continuation',gpu_count=1,gpu_type='H100',max_seconds=50400,approving_person='unit-test-fixture',approved_at='fixture',source_evidence_file='recovery_r2/fixture_evidence.txt',source_evidence_sha256=sha(evidence))
        self.approval_path=self.recovery/'fixture_approval.json'
        self.plan=dict(status='prepared_not_submitted',parent_sha256=PARENT_SHA,completed_gpu_seconds_upper_bound=35292,remaining_forecast_seconds=35000,checkpoint_sha256=sha(self.cp),journal_sha256=sha(self.run/'training.jsonl'),cpu_restore_proof='recovery_r2/RESTORE_READY.json',terminal_receipt='recovery_r2/PLATFORM_STATUS.json')
        self.plan_path=self.recovery/'plan.json'
    def cleanup(self):
        Path(self.tmp.name).resolve().relative_to(HERE)
        self.tmp.cleanup()
    def check(self,max_seconds=50400):
        self.plan['cpu_restore_proof_sha256']=dump(self.proof_path,self.proof)
        self.plan['terminal_receipt_sha256']=dump(self.terminal_path,self.terminal)
        return validate_resume(self.root,self.cp,self.plan_path,dump(self.plan_path,self.plan),self.approval_path,dump(self.approval_path,self.approval),max_seconds)
    def test_valid_fixture_gate_only(self):self.assertEqual(self.check()['next_update'],401)
    def test_pending_approval_rejected(self):
        self.approval['status']='pending'
        with self.assertRaisesRegex(ValueError,'approval'):self.check()
    def test_wrong_authority_rejected(self):
        self.approval['authority']='experiment_operator'
        with self.assertRaisesRegex(ValueError,'approval'):self.check()
    def test_missing_approval_source_rejected(self):
        self.approval['source_evidence_file']='recovery_r2/missing.txt'
        with self.assertRaises(FileNotFoundError):self.check()
    def test_modified_journal_rejected(self):
        (self.run/'training.jsonl').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'journal'):self.check()
    def test_modified_checkpoint_rejected(self):
        self.cp.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint'):self.check()
    def test_per_attempt_cap_rejected(self):
        with self.assertRaisesRegex(ValueError,'14 hours'):self.check(50401)
    def test_cumulative_budget_rejected(self):
        self.plan['completed_gpu_seconds_upper_bound']=40000
        with self.assertRaisesRegex(ValueError,'24 GPU'):self.check()
    def test_forecast_does_not_fit_rejected(self):
        self.plan['remaining_forecast_seconds']=50400
        with self.assertRaisesRegex(ValueError,'forecast'):self.check()
    def test_stale_lock_rejected(self):
        (self.run/'.running').write_text('old lock',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'stale lock'):self.check()
    def test_nonterminal_parent_rejected(self):
        self.terminal['status']='RUNNING'
        with self.assertRaisesRegex(ValueError,'not terminal'):self.check()
    def test_unaligned_resume_rejected(self):
        self.proof['progress']['update']=399
        with self.assertRaisesRegex(ValueError,'Unverified'):self.check()

if __name__=='__main__':
    ast.parse((HERE/'job_resume_r2.py').read_text(encoding='utf-8'))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests))
    receipt=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),scope='Temporary CPU fixture tests only; not administrator approval, GPU continuation, or final acceptance',sources={n:sha(HERE/n) for n in ('resume_guard.py','job_resume_r2.py','test_resume_guard.py')})
    with (HERE/'CPU_RESUME_GUARD_TESTS.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2);f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)

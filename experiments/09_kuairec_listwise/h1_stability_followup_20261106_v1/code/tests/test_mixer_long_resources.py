"""Allocation accounting against actual released platform receipts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('resource_audit',ROOT/'scripts/analysis/audit_mixer_long_resources.py')
resource=importlib.util.module_from_spec(spec);spec.loader.exec_module(resource)
FIXTURES=ROOT/'tests/fixtures/stage05_resources'


def inventory():
    return dict(campaign_id=resource.CAMPAIGN,quota_file='PRE_GPU_QUOTA.json',jobs=[
        dict(job_name='nancy-stage05-long-20261020-smoke-r1',mode='smoke',status_file='SMOKE_STATUS_AFTER_EXIT.json',
             event_files=['SMOKE_EVENTS_AFTER_EXIT.json','SMOKE_RELEASE_EVENTS.json']),
        dict(job_name='nancy-stage05-long-no-mixer-frozen-20261020-r1',mode='train_arm',
             status_file='NO_MIXER_FROZEN_R1_TERMINAL_STATUS.json',event_files=['NO_MIXER_FROZEN_R1_EVENTS.json'])])


class ResourceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        provenance=resource.read(FIXTURES/'PROVENANCE.json')
        for name,digest in provenance['files'].items():
            assert hashlib.sha256((FIXTURES/name).read_bytes()).hexdigest()==digest

    def test_actual_smoke_and_failed_attempt_sum_to_682_gpu_seconds(self):
        result=resource.audit(FIXTURES,inventory(),final=False)
        self.assertEqual(result['actual_allocated_gpu_seconds'],682)
        self.assertEqual(result['acceptance_allocated_gpu_seconds'],636)
        self.assertEqual([r['allocated_gpu_seconds'] for r in result['jobs']],[636,46])
        self.assertFalse(result['all_goal_jobs_released'])
        self.assertFalse(result['actual_goal_invoice_settlement_verified'])
        self.assertAlmostEqual(result['estimated_points_from_allocated_time'],682/3600)

    def test_successful_exit_without_pod_deletion_is_not_release(self):
        value=inventory()['jobs'][0];value['event_files']=['SMOKE_EVENTS_AFTER_EXIT.json']
        with self.assertRaisesRegex(ValueError,'deletion proof'):
            resource.released_job(FIXTURES,value)

    def test_duplicate_job_cannot_be_counted_twice(self):
        value=inventory();value['jobs'].append(copy.deepcopy(value['jobs'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate allocation'):
            resource.audit(FIXTURES,value,final=False)

    def test_running_status_does_not_become_released_from_old_deletion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shutil.copytree(FIXTURES,root,dirs_exist_ok=True)
            path=root/'SMOKE_STATUS_AFTER_EXIT.json';data=resource.read(path);data['data']['status']='RUNNING'
            path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'terminal platform status'):
                resource.audit(root,inventory(),final=False)

    def test_final_platform_listing_cannot_omit_an_active_campaign_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shutil.copytree(FIXTURES,root,dirs_exist_ok=True)
            items=[dict(name=r['job_name'],status=s) for r,s in zip(inventory()['jobs'],('SUCCEEDED','FAILED'))]
            items.append(dict(name='nancy-stage05-long-mixer-joint-20261020-r1',status='RUNNING'))
            (root/'listing.json').write_text(json.dumps(dict(success=True,data=dict(items=items))),encoding='utf-8')
            value=inventory();value['final_job_list_file']='listing.json'
            with self.assertRaisesRegex(ValueError,'may be omitted'):
                resource.audit(root,value,final=True)

    def test_partial_training_release_cannot_claim_final_evaluation_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);shutil.copytree(FIXTURES,root,dirs_exist_ok=True)
            items=[dict(name=r['job_name'],status=s) for r,s in zip(inventory()['jobs'],('SUCCEEDED','FAILED'))]
            (root/'listing.json').write_text(json.dumps(dict(success=True,data=dict(items=items))),encoding='utf-8')
            value=inventory();value['final_job_list_file']='listing.json'
            with self.assertRaisesRegex(ValueError,'No successful formal evaluation'):
                resource.audit(root,value,final=True)


if __name__=='__main__':unittest.main()

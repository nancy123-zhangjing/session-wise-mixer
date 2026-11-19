import json
import tempfile
import unittest
import shutil
from pathlib import Path
from scripts.session_wise_ar.report import report
from scripts.rqcd_mixer.common import sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ReportTest(unittest.TestCase):
    def test_cf_native_schema_and_wrong_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); train = root/'train'; train.mkdir()
            checkpoint = train/'last-model.pt'; checkpoint.write_bytes(b'synthetic model receipt fixture')
            digest = sha256(checkpoint)
            common = {'count': 2, 'seconds': 1,
                      'metrics': {'ndcg@5': .01, 'recall@5': .02, 'hr@5': .1}}
            # Unchanged CF engine stores split INSIDE identity, unlike new AR.
            val = dict(common, identity={'checkpoint_sha256': digest, 'split': 'validation', 'R': 35, 'T': 300})
            test = dict(common, identity={'checkpoint_sha256': digest, 'split': 'test', 'R': 35, 'T': 300})
            write(train/'TRAINING_COMPLETE.json', {'arm':'B0', 'R':35, 'final_checkpoint':'last-model.pt', 'final_sha256':digest})
            write(train/'BEST.json', {'R':35, 'checkpoint':'last-model.pt', 'sha256':digest})
            write(train/'TEST_COMPLETE.json', {'best':test, 'final':test})
            cfg = {'target':{'mode':'cf'}, 'split_counts':{'train':709443}, 'training':{'effective_batch':1024}}
            write(train/'config.json', cfg)
            write(train/'binding.json', {'prepared_sha256':'shared', 'teacher_sha256':'teacher', 'data':{'dataset':{'id':'fixture'}}})
            write(train/'validation/R35/summary.json', val)
            (train/'events.jsonl').write_text(json.dumps({'kind':'validation','R':35,'seconds':1})+'\n')
            rows = report([{'arm':'CF-M0','train':str(train)}], root/'report')
            self.assertEqual(rows[0]['r_updates'], 24255)
            self.assertEqual(rows[0]['t_updates'], 300)
            self.assertEqual(rows[0]['test_ndcg@5'], .01)
            wrong = dict(test, identity=dict(test['identity'], checkpoint_sha256='wrong'))
            write(train/'TEST_COMPLETE.json', {'best':wrong, 'final':test})
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                report([{'arm':'CF-M0','train':str(train)}], root/'bad')
            cfg['target']['mode'] = 'hybrid'; write(train/'config.json',cfg)
            with self.assertRaisesRegex(ValueError, 'requires original CF'):
                report([{'arm':'CF-M0','train':str(train)}], root/'bad')

    def test_ar_best_final_and_missing_arm_disclosure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); train, testroot = root/'train', root/'test'
            checkpoint = train/'checkpoints/step_0000004.pt'; checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'artificial checkpoint')
            digest = sha256(checkpoint)
            common = {'count':2, 'seconds':1, 'identity':{'checkpoint_sha256':digest},
                      'metrics':{'ndcg@5':.02,'recall@5':.03,'hr@5':.1}}
            val, test = dict(common, split='validation'), dict(common, split='test')
            node = {'step':4,'session_epochs':1.25,'r_updates':4,'t_updates':4,'train_seconds':2,
                    'validation':val,'loss':{'total':1},'checkpoint_sha256':digest}
            done = {'arm':'AR-D1', 'parameter_count':100,'progress':{'done':True,'train_seconds':2,'records':[node]},
                    'best':node,'final':node, 'prepared_sha256':'shared', 'input_binding':{'dataset':{'id':'fixture'}}}
            write(train/'complete.json',done); write(testroot/'complete.json',{'arm':'AR-D1','best':test,'final':test})
            rows = report([{'arm':'AR-D1','train':str(train),'test':str(testroot)}],root/'report')
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['session_epochs'],1.25)
            self.assertIn('Missing arms: AR-T0', (root/'report/REPORT.md').read_text())
            other, other_test = root/'other_train', root/'other_test'
            shutil.copytree(train, other); shutil.copytree(testroot, other_test)
            other_done = dict(done, arm='AR-D0', prepared_sha256='different_parent')
            write(other/'complete.json', other_done)
            write(other_test/'complete.json', {'arm':'AR-D0','best':test,'final':test})
            entries = [{'arm':'AR-D1','train':str(train),'test':str(testroot)},
                       {'arm':'AR-D0','train':str(other),'test':str(other_test)}]
            with self.assertRaisesRegex(ValueError, 'same shared parent'):
                report(entries,root/'bad')
            other_done.update(prepared_sha256='shared', input_binding={'dataset':{'id':'wrong-split'}})
            write(other/'complete.json',other_done)
            with self.assertRaisesRegex(ValueError, 'identical dataset'):
                report(entries,root/'bad')


if __name__ == '__main__': unittest.main()

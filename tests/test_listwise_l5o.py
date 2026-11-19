"""L5O adapter, execution optimizations and legacy compatibility on real modules."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from tests.test_listwise_assets import make_dataset, make_assets, rebind_dataset
from scripts.listwise_diffgrm.data import KuaiRecListDataset
from scripts.listwise_diffgrm.execution import TrainingTensorCache
from scripts.listwise_diffgrm.run import build_context, evaluate, phase_indices
from scripts.listwise_diffgrm.training import phase_plan, rng_state, restore_rng, cpu_copy
from scripts.listwise_diffgrm.decoding import generate_lists
from scripts.listwise_diffgrm.run import make_catalog

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class L5OIntegration(unittest.TestCase):
    def setup_fixture(self, root):
        data = make_dataset(root/'data')
        cfg = json.loads((data/'config.json').read_text())
        cfg.update(protocol_id='kuairec_l5o_records_v1', allow_repeated_targets=True)
        (data/'config.json').write_text(json.dumps(cfg))
        for split in ('train','validation','test'):
            rows = pq.read_table(data/f'{split}.parquet').to_pylist()
            for i, row in enumerate(rows):
                row['target_item_ids'][1] = row['target_item_ids'][0]
                row['target_watch_ratio'] = [0]*5 if i == 0 else [1,0,0,2,0]
            pq.write_table(pa.Table.from_pylist(rows), data/f'{split}.parquet', row_group_size=1)
        rebind_dataset(data)
        assets = make_assets(data,root/'assets')
        c = json.loads((ROOT/'configs/09_kuairec_listwise/l5o.json').read_text())
        c.update(dataset=str(data), assets=str(assets.root), history_max=7)
        c['model'].update(codebook_size=8,n_embd=16,n_inner=32)
        c['training'].update(effective_batch_size=2,microbatch_size=1,r_updates_per_unit=1,
            warmup_epochs=1,cycles=1,r_epochs_per_cycle=1,t_updates_per_cycle=1,decoder_warmup_updates=1)
        c['evaluation'].update(val_beam=2,test_beam=2,candidate_lists=2,batch_size=2)
        return c

    def test_vectorized_cache_matches_every_field_with_repeats_and_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.setup_fixture(Path(tmp))
            data = KuaiRecListDataset(cfg['dataset'],'train')
            a,b = TrainingTensorCache(data), TrainingTensorCache(data,vectorized=True)
            for idx in ([2,0,1,2],[1],[]):
                for k in a.fields:
                    self.assertTrue(torch.equal(a.select(idx)[k],b.select(idx)[k]))
            data.table = data.table.slice(1,2)
            a,b = TrainingTensorCache(data), TrainingTensorCache(data,vectorized=True)
            for k in a.fields:
                self.assertTrue(torch.equal(a.select([1,0])[k],b.select([1,0])[k]))

    def test_metrics_batching_beam_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cfg=self.setup_fixture(root)
            trainer,assets,data,binding=build_context(cfg,'cpu','fp32',True)
            rng=cpu_copy(rng_state())
            a=evaluate(trainer,assets,split='validation',destination=root/'a')
            self.assertTrue(torch.equal(rng['torch'],rng_state()['torch']))
            cfg['evaluation']['batch_size']=1
            cfg['execution']['batched_frontier']=False
            b=evaluate(trainer,assets,split='validation',destination=root/'b')
            self.assertEqual(a['metrics'],b['metrics'])
            self.assertEqual(a['metrics']['ev_samples'],2)
            with gzip.open(root/'a/per_sample.jsonl.gz','rt') as f:
                rows=[json.loads(s) for s in f]
            self.assertIsNone(rows[0]['metrics']['ev_ndcg@5'])

    def test_reference_and_fast_stochastic_rt_updates(self):
        for mixer in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                cfg=self.setup_fixture(Path(tmp))
                fast,_,data,_=build_context(cfg,'cpu','fp32',mixer)
                slow_cfg=copy.deepcopy(cfg);slow_cfg['execution']={}
                slow,_,_,_=build_context(slow_cfg,'cpu','fp32',mixer)
                batch=data.collator()([data[0],data[1]])
                for kind in ('R','T'):
                    rng=rng_state()
                    a=fast.update([batch],kind)
                    restore_rng(rng)
                    b=slow.update([batch],kind)
                    self.assertEqual(a,b)
                    for name,value in fast.model.state_dict().items():
                        self.assertTrue(torch.equal(value,slow.model.state_dict()[name]))
                    for name,value in fast.alignment.state_dict().items():
                        self.assertTrue(torch.equal(value,slow.alignment.state_dict()[name]))

    def test_budget_is_old_updates_not_new_dataset_epochs(self):
        cfg=json.loads((ROOT/'configs/09_kuairec_listwise/l5o.json').read_text())
        plan=phase_plan(cfg,2300000)
        self.assertEqual(sum(x['updates'] for x in plan if x['kind']=='R'),24255)
        self.assertEqual(sum(x['updates'] for x in plan if x['kind']=='T'),400)
        self.assertEqual(sum(x['validate'] for x in plan),35)
        ids=phase_indices(cfg,2300000,0,plan[0])
        self.assertEqual(len(ids),693*1024)
        self.assertEqual(len(np.unique(ids)),len(ids))


if __name__ == '__main__':
    unittest.main()

"""Synthetic checks only: never use real datasets, checkpoints or remote GPUs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from scripts.rqcd_mixer.config import load_config, TARGET_CONFIGS

import torch
from torch import nn
from scripts.rqcd_mixer.common import seed_all, cpu_copy, atomic_save
from scripts.rqcd_mixer.targets import (target_inputs, collaborative_loss, teacher_spec, teacher_sources, load_teacher)
from scripts.rqcd_mixer.teacher import next_item_batch, TeacherTrainer
from scripts.rqcd_mixer.training import Pretrainer
from scripts.rqcd_mixer.losses import landing_loss
from scripts.rqcd_mixer.tokenizer import distances
from test_rqcd_mixer import tiny_config, cpu_runtime, SyntheticData, assert_tree_equal, fixture

torch.set_num_threads(1)


def target_config(mode='hybrid'):
    cfg = tiny_config()
    full = load_config(TARGET_CONFIGS[mode])
    for key in ('target', 'teacher', 'loss'):
        cfg[key] = copy.deepcopy(full[key])
    cfg['design_sha256'] = full['design_sha256']
    cfg['teacher'].update(width=8, layers=2, heads=2, ffn=16, batch_size=4, dropout=.1)
    cfg['target']['teacher_dim'] = 8
    return cfg


class ProtocolTests(unittest.TestCase):
    def test_only_approved_settings_differ(self):
        old = load_config()
        for path in TARGET_CONFIGS.values():
            cfg = load_config(path)
            unchanged = set(old) - {'protocol_id', 'design_sha256', 'loss'}
            for key in unchanged:
                self.assertEqual(cfg[key], old[key], key)
            self.assertEqual(cfg['loss']['landing'], .1)
            self.assertEqual(cfg['target']['selection'], 'objective')
        self.assertEqual(load_config(TARGET_CONFIGS['cf'])['teacher'],
                         load_config(TARGET_CONFIGS['hybrid'])['teacher'])

    def test_scientific_override_rejected(self):
        cfg = load_config(TARGET_CONFIGS['cf'])
        cfg['training']['r_lr'] *= 2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                load_config(path)


class TeacherTests(unittest.TestCase):
    def test_prefix_windows_never_include_current_or_future_item(self):
        cfg = target_config(); data = SyntheticData(cfg)
        ids = torch.arange(5)
        history, label = next_item_batch(data, ids, cfg, 'cpu')
        for j in range(5):
            expected = torch.cat((data.history[0], data.target[0, :j]))[-cfg['history_max']:]
            self.assertTrue(torch.equal(history[j], expected))
            self.assertEqual(label[j], data.target[0, j])
        altered = copy.deepcopy(data); altered.target[0, 2:] += 100
        before, _ = next_item_batch(data, torch.tensor([2]), cfg, 'cpu')
        after, _ = next_item_batch(altered, torch.tensor([2]), cfg, 'cpu')
        self.assertTrue(torch.equal(before, after))

    def test_teacher_resume_and_left_padding(self):
        cfg = target_config(); data = SyntheticData(cfg); rt = cpu_runtime()
        seed_all(13)
        a = TeacherTrainer(cfg, rt, data, 'cpu', {'fixture': 1})
        self.assertTrue(torch.isfinite(a.model(data.history)).all())
        a.advance(); saved = a.state(); a.advance(); expected = a.state()
        b = TeacherTrainer(cfg, rt, data, 'cpu', {'fixture': 1}); b.restore(saved); b.advance()
        assert_tree_equal(self, expected, b.state())
        self.assertTrue(torch.equal(a.model.items.weight[0], torch.zeros(8)))

    def test_feature_scaling_and_content_duplicate_mask(self):
        cfg = target_config(); content = torch.randn(12, 8); content[1] = content[0]
        teacher = {'embedding': torch.randn(12, 8), 'train_supported': torch.ones(12, dtype=torch.bool)}
        hybrid = target_inputs(content, teacher, cfg)
        self.assertTrue(torch.equal(hybrid['features'], content))
        self.assertEqual(hybrid['groups'][0], hybrid['groups'][1])
        cfg['target']['mode'] = 'cf'; cf = target_inputs(content, teacher, cfg)
        self.assertAlmostEqual(float(cf['features'].square().sum(-1).mean()), 1., places=6)
        self.assertTrue(torch.allclose(cf['features']*cf['cf_global_scale'], teacher['embedding']))
        q = torch.randn(2, 8, requires_grad=True); projection = nn.Linear(8, 8)
        loss = collaborative_loss(q, teacher['embedding'][:2], projection,
                                  torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.bool), .1)
        self.assertEqual(float(loss), 0.)
        loss = collaborative_loss(q, teacher['embedding'][:2], projection,
                                  torch.arange(2), torch.zeros(2, dtype=torch.bool), .1)
        self.assertEqual(float(loss), 0.); loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_teacher_asset_rejects_wrong_item_order(self):
        cfg = target_config(); ids = torch.arange(12)
        asset = {'format': 'rqcd_teacher_items_v1', 'spec': teacher_spec(cfg), 'data_binding': {'fixture': 1},
                 'source': teacher_sources(), 'original_ids': ids, 'embedding': torch.randn(12, 8),
                 'train_supported': torch.ones(12, dtype=torch.bool)}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'teacher.pt'; atomic_save(p, asset)
            load_teacher(p, cfg, {'fixture': 1}, ids.numpy())
            with self.assertRaises(ValueError):load_teacher(p, cfg, {'fixture': 1}, ids.flip(0).numpy())
            with self.assertRaises(ValueError):load_teacher(p, cfg, {'fixture': 2}, ids.numpy())


class TargetPretrainingTests(unittest.TestCase):
    def test_hybrid_alignment_reaches_encoder_but_not_centers_directly(self):
        cfg, content, joint = fixture(cfg=target_config())
        model = joint.tokenizer; model.requires_grad_(True)
        teacher = torch.randn(len(content), 8, requires_grad=True)
        losses = model.pretrain_forward(content, teacher, torch.arange(len(content)),
                                        torch.ones(len(content), dtype=torch.bool))
        losses['cf'].backward()
        self.assertGreater(float(sum(p.grad.abs().sum() for p in model.encoder.parameters())), 0)
        self.assertIsNone(model.books.grad)
        self.assertTrue(all(p.grad is None for p in model.mixer.parameters()))
        self.assertIsNone(teacher.grad)

    def test_pretraining_resume_and_fixed_objective(self):
        for mode in ('cf', 'hybrid'):
            cfg, content, joint = fixture(cfg=target_config(mode))
            inputs = target_inputs(content, {'embedding': torch.randn(len(content), 8),
                                  'train_supported': torch.ones(len(content), dtype=torch.bool)}, cfg)
            joint.tokenizer.requires_grad_(True)
            a = Pretrainer(cfg, cpu_runtime(), joint.tokenizer, inputs['features'], {'toy': 1}, inputs)
            a.advance(); saved = a.state()
            before = a.stream.state(); first = a.evaluate_objective(); second = a.evaluate_objective()
            self.assertEqual(first, second); assert_tree_equal(self, before, a.stream.state())
            a.advance(); expected = a.state()
            _, _, fresh = fixture(cfg=cfg); fresh.tokenizer.requires_grad_(True)
            b = Pretrainer(cfg, cpu_runtime(), fresh.tokenizer, inputs['features'], {'toy': 1}, inputs)
            b.restore(saved); b.advance(); assert_tree_equal(self, expected, b.state())


class LandingTests(unittest.TestCase):
    def test_matches_independent_item_enumeration_with_collision_and_exclusion(self):
        cfg, _, trainer = fixture(cfg=target_config())
        catalog = trainer.catalog
        # Identical latent rows force a real SID collision, with tied bucket distances.
        asset = catalog.state(); asset['z'] = asset['z'].clone(); asset['z'][1] = asset['z'][0]
        from scripts.rqcd_mixer.tokenizer import hard_quantize
        from scripts.rqcd_mixer.catalog import Catalog
        hard = hard_quantize(asset['z'], asset['books']); asset.update(sid=hard['sid'], q=hard['q'])
        catalog = Catalog(asset, 'cpu')
        ids = torch.tensor([[1, 2, 3, 4, 5], [2, 2, 1, 7, 2]])
        pred = catalog.z_pad[ids].clone()+.04*torch.randn(2, 5, cfg['rq']['latent_dim'])
        pred.requires_grad_(True); books = catalog.books.clone().requires_grad_(True)
        actual = landing_loss(pred, ids, books, catalog, cfg)
        paths, buckets = [], []
        for row in range(2):
            for j in range(5):
                item = int(ids[row,j]); sid = catalog.sid_pad[item]; earlier = set(ids[row,:j].tolist()) - {item}
                remaining = [v for v in range(1, catalog.count+1) if v not in earlier]
                residual = pred[row,j]; value = pred.new_zeros(())
                for level, book in enumerate(books):
                    prefix = [v for v in remaining if torch.equal(catalog.sid_pad[v,:level],sid[:level])]
                    codes = sorted(set(int(catalog.sid_pad[v,level]) for v in prefix))
                    logits = -distances(residual[None],book[codes])[0]/(
                        cfg['quantization']['temperature']*catalog.stats['distance_scale'][level])
                    value += torch.logsumexp(logits,0)-logits[codes.index(int(sid[level]))]
                    residual = residual-book[sid[level]]
                bucket = [v for v in remaining if torch.equal(catalog.sid_pad[v],sid)]
                d = distances(pred[row,j][None],catalog.z_pad[bucket])[0]
                logits = -d/(cfg['loss']['bucket_temperature']*2*catalog.dim*catalog.stats['scale'].square())
                paths.append(value);buckets.append(torch.logsumexp(logits,0)-logits[bucket.index(item)])
        self.assertTrue(torch.allclose(actual['path'],torch.stack(paths).mean(),atol=2e-5))
        self.assertTrue(torch.allclose(actual['bucket'],torch.stack(buckets).mean(),atol=2e-5))
        self.assertTrue(torch.allclose(actual['landing'],actual['path']+actual['bucket']))
        actual['landing'].backward()
        self.assertTrue(torch.isfinite(pred.grad).all());self.assertGreater(float(pred.grad.norm()),0)
        self.assertTrue(torch.isfinite(books.grad).all());self.assertGreater(float(books.grad.norm()),0)

    def test_joint_freezing_microbatch_and_partial_T_resume(self):
        for mode in ('cf','hybrid'):
            cfg, _, a = fixture(cfg=target_config(mode), micro=2)
            _, _, full = fixture(cfg=cfg, micro=4)
            data = SyntheticData(cfg); history,target = data.batch(torch.arange(4),'cpu')
            before = cpu_copy(a.tokenizer.state_dict())
            draws = a.draws(4,'R')
            a.update(history,target,'R',draws=draws);full.update(history,target,'R',draws=draws)
            for k,v in a.model.state_dict().items():
                self.assertTrue(torch.allclose(v,full.model.state_dict()[k],atol=2e-6,rtol=2e-5),k)
            assert_tree_equal(self,before,a.tokenizer.state_dict())
            draws = a.draws(4,'T')
            a.update(history,target,'T',draws=draws);full.update(history,target,'T',draws=draws)
            for k,v in a.tokenizer.state_dict().items():
                self.assertTrue(torch.allclose(v,full.tokenizer.state_dict()[k],atol=2e-6,rtol=2e-5),k)
            self.assertFalse(torch.equal(a.tokenizer.books, before['books']))
            # Use advance for consistent stream/update/phase clocks in resume state.
            _, _, a = fixture(cfg=cfg, micro=2)
            while a.phase != 'T':a.advance(data)
            frozen = cpu_copy(a.model.state_dict()); encoder = cpu_copy(a.tokenizer.encoder.state_dict())
            decoder = cpu_copy(a.tokenizer.decoder.state_dict())
            a.advance(data);saved=a.state();a.advance(data);expected=a.state()
            _, _, b = fixture(cfg=cfg,micro=2);b.restore(saved);b.advance(data)
            assert_tree_equal(self,expected,b.state())
            assert_tree_equal(self,frozen,a.model.state_dict());assert_tree_equal(self,encoder,a.tokenizer.encoder.state_dict())
            assert_tree_equal(self,decoder,a.tokenizer.decoder.state_dict())
            self.assertIn('landing',a.summary('T'));self.assertNotIn('align',a.summary('T'))
            self.assertTrue(all(p.grad is None for p in a.tokenizer.encoder.parameters()))


class PipelineTests(unittest.TestCase):
    def test_teacher_entry_selects_validation_best_without_reading_test(self):
        from scripts.rqcd_mixer.teacher import train_teacher
        from scripts.rqcd_mixer.common import read_state
        import contextlib
        import io
        cfg=target_config();cfg['teacher'].update(epochs=3,patience=1)
        seen=[]
        def data(path,split,cfg,device):
            seen.append(split);return SyntheticData(cfg,split)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch('scripts.rqcd_mixer.run._runtime',return_value=(cpu_runtime(),{'toy':1})), \
             patch('scripts.rqcd_mixer.teacher.dataset_binding',return_value={'history_max':3,'list_length':5}), \
             patch('scripts.rqcd_mixer.teacher.catalog_original_ids',return_value=np.arange(12)), \
             patch('scripts.rqcd_mixer.teacher.PackedSplit',side_effect=data), \
             patch.object(TeacherTrainer,'evaluate',side_effect=[.2,.1]):
            args=SimpleNamespace(device='cpu',dataset=Path(tmp)/'not_real',output=Path(tmp)/'teacher',resume=False,runtime=None)
            result=train_teacher(args,cfg)
            self.assertEqual((result['selected_epoch'],result['completed_epochs']),(1,2))
            exported=read_state(args.output/'items.pt');best=read_state(args.output/'best.pt')
            self.assertTrue(torch.equal(exported['embedding'],best['model']['items.weight'][1:]))
            args.resume=True;self.assertEqual(train_teacher(args,cfg),result)
            self.assertEqual(seen,['train','validation','train','validation'])

    def test_two_target_pipeline_selection_resume_binding_and_report(self):
        """Tiny in-memory data only: exercise actual prepare/pretrain/train/test I/O."""
        from scripts.rqcd_mixer import run
        from scripts.rqcd_mixer.common import read_state
        from scripts.rqcd_mixer.report import report
        import contextlib
        import io
        roots = []
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            teacher_path = root/'teacher.pt'
            cfg = target_config('cf')
            atomic_save(teacher_path, {'format':'rqcd_teacher_items_v1', 'spec':teacher_spec(cfg),
                        'data_binding':{'toy':1}, 'source':teacher_sources(), 'original_ids':torch.arange(12),
                        'embedding':torch.randn(12,8), 'train_supported':torch.ones(12,dtype=torch.bool)})
            content = torch.nn.functional.normalize(torch.randn(12,8),dim=-1).numpy()
            content_path = root/'content.npy';np.save(content_path,content)
            with patch.object(run,'_runtime',return_value=(cpu_runtime(),{'toy':1})), \
                 patch('scripts.rqcd_mixer.data.bind_inputs',return_value={'dataset':{'toy':1},'content_sha256':'toy'}), \
                 patch('scripts.rqcd_mixer.data.catalog_original_ids',return_value=np.arange(12)), \
                 patch('scripts.rqcd_mixer.data.PackedSplit',side_effect=lambda path,split,cfg,device: SyntheticData(cfg,split)):
                for mode in ('cf','hybrid'):
                    cfg = target_config(mode)
                    args=SimpleNamespace(device='cpu',dataset=root/'not_real',content=content_path,teacher=teacher_path,
                         output=root/(mode+'_prepared'),runtime=None,arm='B1',resume=False)
                    run.prepare(args,cfg);args.prepared=args.output
                    args.output=root/(mode+'_rq')
                    selected=run.pretrain(args,cfg)
                    self.assertEqual(selected['selected_checkpoint'],'best_objective.pt')
                    self.assertIn(selected['best_objective_epoch'],[1,2])
                    args.pretrained=args.output;args.output=root/mode
                    run.train(args,cfg);args.run=args.output;run.test(args,cfg)
                    state=read_state(args.run/'last.pt')
                    self.assertEqual((state['r_epoch'],state['t_updates']),(3,4))
                    args.resume=True;run.train(args,cfg)
                    resumed=read_state(args.run/'last.pt')
                    assert_tree_equal(self,state,resumed)
                    roots.append(args.run)
                self.assertNotEqual(read_state(root/'cf_rq/catalog.pt')['config'],
                                    read_state(root/'hybrid_rq/catalog.pt')['config'])
                cfmeta=json.loads((root/'cf_prepared/PREPARED.json').read_text())
                hmmeta=json.loads((root/'hybrid_prepared/PREPARED.json').read_text())
                self.assertEqual(cfmeta['teacher_sha256'],hmmeta['teacher_sha256'])
                self.assertEqual(cfmeta['common_recommender_hash'],hmmeta['common_recommender_hash'])
                args.prepared=root/'cf_prepared'
                with self.assertRaises(ValueError):run._prepared(args,target_config('hybrid'))
                args.prepared=root/'hybrid_prepared'
                original=teacher_path.read_bytes();teacher_path.write_bytes(original+b'changed')
                with self.assertRaises(ValueError):run._prepared(args,target_config('hybrid'))
                teacher_path.write_bytes(original)
            result=report(roots,root/'report')
            self.assertEqual(result['rows'],4)
            rows=json.loads((root/'report/comparison.json').read_text())
            self.assertEqual({r['arm'] for r in rows},{'cf/B1','hybrid/B1'})
            self.assertTrue((root/'report/T_landing.png').exists())

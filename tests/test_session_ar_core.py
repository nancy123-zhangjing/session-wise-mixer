import unittest
import torch
from scripts.session_wise_ar.config import load, plan
from scripts.session_wise_ar.data import compact_context, SessionTasks
from scripts.session_wise_ar.model import ARRecommender


def tiny(arm):
    c = load(arm)
    c.update(catalog_size=12, history_max=3)
    c['rq'].update(input_dim=8, hidden_dims=[16], latent_dim=8, codes=8,
                   kmeans_iterations=2, sinkhorn_iterations=3)
    c['model'].update(width=12, ffn=24, heads=2, d_kv=4,
                      encoder_layers=1, decoder_layers=1, dropout=0.0)
    c['evaluation']['beam'] = 3
    return c


class CoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_prefix_compacts_and_excludes_future(self):
        history = torch.tensor([[0, 1, 2], [0, 0, 3]])
        prefix = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
        result, mask = compact_context(history, prefix, torch.tensor([0, 2]))
        self.assertEqual(result.tolist(), [[1, 2, 0, 0, 0, 0, 0], [3, 8, 9, 0, 0, 0, 0]])
        self.assertEqual(mask.sum(1).tolist(), [2, 3])

    def test_native_budgets_keep_task_batch_semantics(self):
        tiger, diger = plan(load('AR-T0')), plan(load('AR-D0'))
        self.assertEqual(tiger['tasks_per_session_epoch'], 709443*5)
        self.assertEqual(tiger['updates_per_epoch'], 13857)
        self.assertEqual(tiger['max_recommender_updates'], 200000)
        self.assertEqual(diger['max_joint_tokenizer_updates'], 1662840)

    def test_native_cached_device_follows_module_move(self):
        model = ARRecommender(tiny('AR-D0'))
        model.to('meta')
        self.assertEqual(model.native.device, model.native.model.shared.weight.device)
        self.assertEqual(model.native.device.type, 'meta')

    def test_native_forward_and_next_token(self):
        for arm in ('AR-T0', 'AR-D0'):
            c = tiny(arm)
            model = ARRecommender(c).eval()
            sid = torch.randint(0, 8, (13, 4)); sid[0] = -1
            batch = {'context': torch.tensor([[1, 2, 0], [3, 4, 5]]),
                     'target': torch.tensor([6, 7]), 'users': torch.tensor([2, 3])}
            loss = model.hard_loss(batch, sid)
            self.assertTrue(torch.isfinite(loss))
            hidden, mask = model.encode(batch['context'], sid, batch['users'])
            for position in range(4):
                p = model.next_logprobs(hidden, mask, sid[batch['target'], :position])
                self.assertEqual(p.shape, (2, 8))
                self.assertTrue(torch.isfinite(p).all())
                # TIGER keeps probability mass outside this position's vocab.
                if arm == 'AR-T0':
                    self.assertTrue((p.exp().sum(-1) < 1).all())
                else:
                    torch.testing.assert_close(p.exp().sum(-1), torch.ones(2))

    def test_tiger_six_heads_are_not_width_divisors(self):
        c = load('AR-T0')
        model = ARRecommender(c)
        attention = model.native.encoder.block[0].layer[0].SelfAttention
        self.assertEqual(attention.q.out_features, 384)
        self.assertEqual(len(model.native.encoder.block), 4)

    def test_current_item_decoder_cannot_read_future_digits(self):
        for arm in ('AR-T0', 'AR-D0'):
            m = ARRecommender(tiny(arm)).eval()
            sid = torch.randint(0, 8, (13, 4)); sid[0] = -1
            context = torch.tensor([[1, 2]])
            emb, mask = m.history_embeddings(context, sid, torch.tensor([0]))
            left = torch.tensor([[1, 2, 3, 4]])
            right = torch.tensor([[1, 2, 6, 7]])
            if arm == 'AR-T0': left, right = m.global_tokens(left), m.global_tokens(right)
            a = m.native(inputs_embeds=emb, attention_mask=mask, labels=left).logits
            b = m.native(inputs_embeds=emb, attention_mask=mask, labels=right).logits
            torch.testing.assert_close(a[:, :3], b[:, :3], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()

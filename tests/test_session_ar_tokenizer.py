import unittest
import torch
from test_session_ar_core import tiny
from scripts.session_wise_ar.model import ARRecommender
from scripts.session_wise_ar.tokenizer import make_tokenizer, joint_loss


class TokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_mixer_same_pretraining_forward(self):
        x = torch.randn(12, 8)
        for family in ('T', 'D'):
            models = []
            for suffix in ('0', '1'):
                torch.manual_seed(7)
                m = make_tokenizer(tiny('AR-'+family+suffix))
                # Shared artificial centers isolate the Mixer adapter (no kmeans).
                with torch.no_grad():
                    books = m.base_books()
                    vals = torch.arange(books.numel()).reshape_as(books).float()/100
                    if family == 'T': m.books.copy_(vals)
                    else:
                        for dst, src in zip(m.original_weights(), vals): dst.copy_(src)
                    m.initialized.fill_(True)
                models.append(m)
            a, b = models
            torch.testing.assert_close(a.loss(x)[0], b.loss(x)[0], rtol=0, atol=0)
            torch.testing.assert_close(a.indices(x), b.indices(x), rtol=0, atol=0)

    def test_recommendation_gradient_and_frozen_encoder(self):
        cfg = tiny('AR-D1')
        pre = make_tokenizer(cfg)
        pre.initialized.fill_(True)
        tok = make_tokenizer(cfg, joint=True)
        tok.load_pretrained(pre)
        rec = ARRecommender(cfg)
        with torch.no_grad(): rec.native.semantic_embedding.weight.normal_()
        sid = torch.randint(0, 8, (13, 4)); sid[0] = -1
        batch = {'context': torch.tensor([[1, 2, 0], [3, 4, 5]]),
                 'target': torch.tensor([6, 7]), 'users': torch.tensor([2, 3])}
        _, parts = joint_loss(rec, tok, batch, sid)
        (parts['semantic'] + parts['suffix']).backward()
        self.assertTrue(all(p.grad is None for p in tok.native.encoder.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in tok.original_weights()), 0)
        self.assertGreater(tok.mixer.out.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in tok.parameters() if p.grad is not None))

    def test_effective_books_reach_native_export_and_loss(self):
        # A nonzero Mixer is equivalent to installing its effective centers in
        # a plain upstream quantizer, including the hard catalog export path.
        mixed = make_tokenizer(tiny('AR-D1')).eval()
        plain = make_tokenizer(tiny('AR-D0')).eval()
        plain.native.encoder.load_state_dict(mixed.native.encoder.state_dict())
        plain.native.decoder.load_state_dict(mixed.native.decoder.state_dict())
        with torch.no_grad():
            mixed.mixer.out.bias.fill_(0.2)
            for dst, src in zip(plain.original_weights(), mixed.effective_books()): dst.copy_(src)
            mixed.initialized.fill_(True); plain.initialized.fill_(True)
        x = torch.randn(12, 8)
        torch.testing.assert_close(mixed.indices(x), plain.indices(x), atol=0, rtol=0)
        torch.testing.assert_close(mixed.loss(x)[0], plain.loss(x)[0], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()

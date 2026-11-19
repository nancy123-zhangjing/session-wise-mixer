"""Small dependency/runtime test for collision suffixes and strict evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "DiffGRM"))

from genrec.models.AR_GRM.evaluator import AR_GRMEvaluator  # noqa: E402
from genrec.models.AR_GRM.tokenizer import AR_GRMTokenizer  # noqa: E402


def test_collision_suffix() -> None:
    tokenizer = AR_GRMTokenizer.__new__(AR_GRMTokenizer)
    tokenizer.config = {
        "n_digit": 3,
        "codebook_size": 16,
        "require_unique_sid": True,
    }
    tokenizer.semantic_n_digit = 2
    tokenizer.use_collision_suffix = True
    tokenizer.total_n_digit = 3
    tokenizer.sid_offset = 3
    tokenizer.log = lambda *_args, **_kwargs: None
    tokenizer.dataset = SimpleNamespace(item2id={"A": 1, "B": 2, "C": 3})

    semantic_ids = {"A": [4, 9], "B": [4, 9], "C": [7, 2]}
    tokenizer.item2tokens = tokenizer._semantic_ids_to_model_tokens(semantic_ids)
    tokenizer.tokens2item = tokenizer._create_reverse_mapping()
    tokenizer._validate_full_mapping()

    assert tokenizer.semantic_collision_stats["collided_items"] == 2
    assert tokenizer.full_collision_stats["collided_items"] == 0
    assert len(set(tokenizer.item2tokens.values())) == 3


def test_strict_item_evaluator() -> None:
    class FakeTokenizer:
        mapping = {
            (1, 0, 0): 1,
            (2, 0, 0): 2,
            (3, 0, 0): 3,
        }

        def codebooks_to_item_id(self, sequence):
            return self.mapping.get(tuple(sequence))

    evaluator = AR_GRMEvaluator({"topk": [2, 3]}, FakeTokenizer())
    predictions = torch.tensor(
        [
            [[2, 0, 0], [1, 0, 0], [3, 0, 0]],
            [[3, 0, 0], [3, 0, 0], [1, 0, 0]],
        ]
    )
    target_item_ids = torch.tensor([1, 2])
    metrics = evaluator.calculate_metrics(predictions, target_item_ids)

    assert metrics["hit_rate@2"].tolist() == [1.0, 0.0]
    assert metrics["legal@10"].tolist() == [1.0, 1.0]
    assert torch.allclose(
        metrics["duplicate@10"], torch.tensor([0.0, 1.0 / 3.0])
    )


if __name__ == "__main__":
    test_collision_suffix()
    test_strict_item_evaluator()
    print("Collision suffix and strict item evaluator: OK")


"""Small CPU test for Cross-PSE tensor shapes, masks, and gradients."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "external" / "DiffGRM" / "genrec" / "models" / "DIFF_GRM" / "cross_pse.py"
)
spec = importlib.util.spec_from_file_location("cross_pse_standalone", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
CrossPSECodebookMixer = module.CrossPSECodebookMixer


def main() -> None:
    torch.manual_seed(7)
    base = torch.randn(4, 16, 8)
    items = torch.randn(11, 4, 8)
    for scope in sorted(CrossPSECodebookMixer.VALID_SCOPES):
        model = CrossPSECodebookMixer(
            base, n_heads=2, attention_scope=scope, residual_scale_init=0.1
        )
        crossed, attention = model.crossed_codebooks(return_attention=True)
        assert crossed.shape == base.shape
        assert attention.shape == (1, 2, 64, 64)
        assert torch.isfinite(crossed).all() and torch.isfinite(attention).all()
        assert torch.allclose(
            attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=1e-5
        )
        result = model.objective(items, 0.01, 0.1, 0.1)
        assert torch.isfinite(result["loss"])
        result["loss"].backward()
        assert model.base_codebooks.grad is None
        assert model.attention.in_proj_weight.grad is not None
        assert torch.isfinite(model.attention.in_proj_weight.grad).all()

    v2 = CrossPSECodebookMixer(
        base,
        n_heads=2,
        attention_scope="full",
        residual_scale_init=1.0,
        learn_residual_scale=False,
        zero_init_attention_output=True,
    )
    initial, _ = v2.crossed_codebooks(return_attention=False)
    assert torch.equal(initial, base)
    assert "residual_scale" not in dict(v2.named_parameters())
    result = v2.objective(items, 0.01, 0.01, 0.1)
    result["loss"].backward()
    output_gradient = v2.attention.out_proj.weight.grad
    assert output_gradient is not None
    assert torch.isfinite(output_gradient).all()
    assert float(output_gradient.abs().sum()) > 0.0

    refined = CrossPSECodebookMixer(
        base,
        n_heads=2,
        train_base_codebooks=True,
        mixer_enabled=False,
        residual_scale_init=1.0,
        learn_residual_scale=False,
    )
    refined_initial, refined_attention = refined.crossed_codebooks(
        return_attention=True
    )
    assert torch.equal(refined_initial, base)
    assert refined_attention is None
    refined_result = refined.objective(items, 0.01, 0.01, 0.1)
    refined_result["loss"].backward()
    assert refined.base_codebooks.grad is not None
    assert float(refined.base_codebooks.grad.abs().sum()) > 0.0
    assert refined.attention.out_proj.weight.grad is None

    v3 = CrossPSECodebookMixer(
        base,
        n_heads=2,
        attention_scope="full",
        residual_scale_init=1.0,
        learn_residual_scale=False,
        zero_init_attention_output=True,
        train_base_codebooks=True,
        mixer_enabled=True,
    )
    v3_initial, _ = v3.crossed_codebooks(return_attention=False)
    assert torch.equal(v3_initial, base)
    v3_result = v3.objective(items, 0.01, 0.01, 0.1)
    v3_result["loss"].backward()
    assert v3.base_codebooks.grad is not None
    assert float(v3.base_codebooks.grad.abs().sum()) > 0.0
    assert v3.attention.out_proj.weight.grad is not None
    assert float(v3.attention.out_proj.weight.grad.abs().sum()) > 0.0
    print("Cross-PSE module test: OK")


if __name__ == "__main__":
    main()

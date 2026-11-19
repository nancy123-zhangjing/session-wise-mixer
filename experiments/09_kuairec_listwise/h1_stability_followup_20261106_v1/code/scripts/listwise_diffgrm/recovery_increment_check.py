"""Disposable real-parent CPU preflight; outputs cannot resume formally."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .recovery import fingerprint
from .recovery_evaluation import write_json
from .recovery_increment import build_increment, gate_metrics
from .run import microbatches
from .training import atomic_save


def parameters(module):
    return {k: v.detach().cpu().clone() for k, v in module.named_parameters()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent", "dataset", "assets", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    controller, _, data = build_increment(args.parent, "cpu", args.dataset, args.assets,
                                          inspection=True)
    trainer, gate = controller.trainer, controller.increment_gate
    initial = controller.checkpoint()
    assert torch.equal(gate.mixer.crossed_codebooks(False)[0], gate.c0)
    before_sid = trainer.alignment.hard_codes.clone()
    trainer.refresh_catalog()
    assert torch.equal(before_sid, trainer.alignment.hard_codes)
    before_r = fingerprint(parameters(trainer.model))
    values = controller.step(microbatches(data, np.arange(32), 32), "T")
    assert fingerprint(parameters(trainer.model)) == before_r
    assert float(gate.mixer.base_codebooks.grad.abs().sum()) > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in gate.mixer.named_parameters() if n != "base_codebooks")
    assert values["increment_alpha_used"] == 0.1
    assert torch.equal(gate.accepted_delta, 0.1 * (
        gate.previous_raw - gate.anchor["raw_start"].to(gate.c0)))
    saved = controller.checkpoint()
    saved["preflight_only"] = True
    saved["inner"]["preflight_only"] = True
    state_path = output / "preflight_only.pt"
    atomic_save(state_path, saved)
    expected_values = controller.step(microbatches(data, np.arange(32, 64), 32), "T")
    expected = controller.checkpoint()
    # Rebuild from the authenticated real parent, then load the actual disk file.
    other, _, data2 = build_increment(args.parent, "cpu", args.dataset, args.assets,
                                      inspection=True)
    disk = torch.load(state_path, map_location="cpu", weights_only=False)
    try:
        other.restore(disk, exact_rng=False)
    except ValueError:
        pass
    else:
        raise AssertionError("Preflight checkpoint accepted as formal state")
    other.restore(disk, exact_rng=False, allow_preflight=True)
    actual_values = other.step(microbatches(data2, np.arange(32, 64), 32), "T")
    assert actual_values == expected_values
    assert fingerprint(other.checkpoint()) == fingerprint(expected)
    for key in ("r_optimizer", "t_optimizer"):
        assert [g["lr"] for g in saved["inner"][key]["param_groups"]] == [
            g["lr"] for g in initial["inner"][key]["param_groups"]]
    before_t = fingerprint(parameters(other.trainer.alignment))
    before_gate = fingerprint(other.increment_gate.state())
    before_model = fingerprint(parameters(other.trainer.model))
    # Isolation-only disposable R update, deliberately outside the formal cursor.
    other.trainer.update(microbatches(data2, np.arange(64, 96), 32), "R")
    assert fingerprint(parameters(other.trainer.alignment)) == before_t
    assert fingerprint(other.increment_gate.state()) == before_gate
    assert fingerprint(parameters(other.trainer.model)) != before_model
    write_json(output / "RESULT.json", dict(
        status="passed", device="cpu", preflight_only=True,
        parent_sha256=controller.protocol["parent_sha256"],
        protocol=controller.protocol, start_codebook_exact=True,
        start_sid_exact=True, T_R_parameter_isolation=True,
        learning_rates_unchanged=True, disk_restore_next_step_exact=True,
        preflight_formal_resume_rejected=True, gate=gate_metrics(controller.trainer)))
    print("INCREMENT_REAL_PARENT_PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    main()

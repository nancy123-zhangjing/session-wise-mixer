# First GPU gate: cached-device repair

The first combined gate passed its TIGER section but failed at DIGER history embedding construction: upstream Model caches `device=cpu` at construction, which PyTorch does not update when `.cuda()` moves parameters. Its temporary embedding allocation consequently remained on CPU.

The adapter now synchronizes that plain attribute in `_apply`, without editing pinned upstream code or changing parameters, losses, schedules or initial tensors. Two targeted CPU tests passed: module movement using the meta device, and native forward/next-token behavior. This is not a GPU acceptance claim; the second combined gate must still pass.

Original `parents/` and failed `gpu_check/` remain intact. `migrate_device_parent.py` creates `parents_devicefix/`, allowing only the adapter source hash to change, preserves all tensor values, and records the original SHA and source map. Worker and second gate use those new parents. The original package manifest remains historical; these explicit patch files and Git commit supplement it. CF uses its original prepared state and unchanged engine.

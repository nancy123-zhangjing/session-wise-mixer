"""Boundary checks for the actual utilization/budget supervisor guard."""
import ast
import importlib.util
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('async_launcher',root/'job_async_r2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    guard=module.UtilizationGuard()
    assert guard.observe(0,20) is None
    assert guard.observe(8999,39) is None
    assert guard.observe(9000,39)=='sustained_low_utilization_save_and_stop'
    guard=module.UtilizationGuard()
    assert guard.observe(0,20) is None
    assert guard.observe(8999,40) is None
    assert guard.low_since is None
    assert guard.observe(9000,20) is None
    assert guard.observe(17999,20) is None
    guard=module.UtilizationGuard()
    assert guard.observe(0,None) is None
    assert guard.observe(299,None) is None
    assert guard.observe(300,None)=='utilization_unobservable_300s'
    assert guard.observe(301,20) is None and guard.unknown_since is None
    gate=json.loads((root/'ASYNC_RESUME_GATE.json').read_text())
    total=gate['consumed_gpu_seconds_upper_bound']+gate['platform_max_seconds']+gate['release_allowance_seconds']
    assert total==85673 and total<86400
    assert gate['forecast_seconds']<gate['inner_max_seconds']-120
    for name in ('async_resume.py','job_async_r2.py','archive_delivery.py'):
        ast.parse((root/name).read_text(encoding='utf-8'))
    result=dict(status='passed_CPU_async_supervisor_guard',checks=['150min_low_utilization_save_stop',
        '40percent_resets_low_window','300s_unobservable_stops_without_fake_liveness',
        '24h_cumulative_budget_with_release','forecast_fits_reserved_shutdown','three_runtime_sources_parse'],
        launcher_sha256=module.sha(root/'job_async_r2.py'),total_gpu_seconds_reserved=total)
    (root/'CPU_ASYNC_GUARD.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()

"""Start existing official SSH/rtunnel binaries in the current CPU Notebook.

Uses the installed CLI's public-key setup and service-start commands only.
Does not run its package install/correction path or alter shared environments.
"""
import contextlib, datetime, hashlib, io, json, logging
from pathlib import Path
from inspire import InspireClient
from inspire.cli.utils.tunnel_reconnect import load_ssh_public_key_material
from inspire.platform.web.browser_api import rtunnel

report = {"schema": "stage09_transfer_service_recovery_v1",
          "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "notebook": "nancy-stage09-prep-20261027-r1",
          "installs_or_upgrades_packages": False,
          "new_workload_or_gpu": False,
          "official_source_sha256": hashlib.sha256(Path(rtunnel.__file__).read_bytes()).hexdigest()}
logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        commands = rtunnel.build_rtunnel_setup_commands(port=31337, ssh_port=22222,
            ssh_public_key=load_ssh_public_key_material())
        ssh_start = [c for c in commands if c.startswith('if [ ! -f "$OPENSSH_INSTALL_FAILED_FILE" ] && [ -x /usr/sbin/sshd ]')]
        tunnel_start = [c for c in commands if c.startswith('if [ -x "$_RT_BIN" ] && ! ps -ef')]
        assert len(ssh_start) == len(tunnel_start) == 1
        assert commands[8].startswith('_RT_ARCH=') and commands[9].startswith('_RT_BIN=')
        setup = commands[:10] + [
            'test -x /usr/sbin/sshd && test -x "$_RT_BIN"',
            'test ! -f "$OPENSSH_INSTALL_FAILED_FILE"',
            *ssh_start, *tunnel_start,
            'echo STAGE09_EXISTING_SERVICES_START_COMMANDS_COMPLETED']
        command = '\n'.join(['set -e'] + setup)
        report["command_sha256"] = hashlib.sha256(command.encode()).hexdigest()
        with InspireClient(account="sii-shared", allow_browser=False,
                           timeout=15, operation_timeout=70) as client:
            result = client.notebooks.exec(report["notebook"], workspace="CPU资源空间",
                command=command, transport="jupyter", timeout=50, capture=True, max_output_bytes=8192)
        report.update(returncode=result.returncode, completed=result.completed,
                      marker_seen='STAGE09_EXISTING_SERVICES_START_COMMANDS_COMPLETED' in result.stdout,
                      truncated=result.truncated)
    except Exception as error:
        report["exception_type"] = type(error).__name__
    report["commands_completed"] = report.get("completed", False) and report.get("returncode") == 0 and report.get("marker_seen", False)
with Path(__file__).with_name("CPU_TRANSFER_SERVICE_RECOVERY_20261028.json").open("x", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(0 if report["commands_completed"] else 1)

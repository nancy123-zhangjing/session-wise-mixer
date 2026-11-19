"""Verify canonical Jupyter SSH route before saving the selected account bridge."""
import contextlib, datetime, io, json, logging, re
from pathlib import Path
from inspire import InspireClient
from inspire.platform.web.browser_api.jupyter_terminal import _notebook_jupyter_url
from inspire.platform.web.browser_api.playwright_notebooks import build_jupyter_proxy_url
from inspire.bridge.tunnel import BridgeProfile, load_tunnel_config, save_tunnel_config, run_ssh_command
from inspire.cli.commands.notebook.notebook_ssh_flow import _identity_file_for_pubkey

name="nancy-stage09-prep-20261027-r1"
report={"schema":"stage09_canonical_ssh_recovery_v1",
        "at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "account":"sii-shared","notebook":name,"command":"hostname",
        "cache_saved":False,"new_remote_services":False}
logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        with InspireClient(account="sii-shared",allow_browser=False,timeout=15,operation_timeout=60) as client:
            nb=client.notebooks.get(name,workspace="CPU资源空间")
            assert nb.status.upper()=="RUNNING"
            with client._transport.scope(timeout=20):
                url=_notebook_jupyter_url(client._transport.session,nb.ref.key)
            assert url
            proxy=build_jupyter_proxy_url(url,port=31337)
            cfg=load_tunnel_config(account="sii-shared")
            existing=cfg.get_bridge(name)
            bridge=BridgeProfile(name=name,proxy_url=proxy,ssh_user="root",ssh_port=22222,
                notebook_id=nb.ref.key,notebook_name=name,workspace_id=nb.ref.workspace_id,
                workspace_name="CPU资源空间",
                identity_file=existing.identity_file if existing else _identity_file_for_pubkey(),rtunnel_port=31337)
            cfg.add_bridge(bridge)
            result=run_ssh_command("hostname",bridge_name=name,config=cfg,timeout=20,capture_output=True,pass_stdin=True)
            hostname=result.stdout.strip()
            valid=bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,252}",hostname))
            report.update(returncode=result.returncode,hostname=hostname if valid else None,
                          argv_command_mode=True,
                          command_not_found="command not found" in (result.stderr or ""),
                          handshake_failure="handshake" in (result.stderr or "").lower())
            if result.returncode==0 and valid:
                save_tunnel_config(cfg)
                report["cache_saved"]=True
    except Exception as error:
        report["exception_type"]=type(error).__name__
with Path(__file__).with_name("CANONICAL_SSH_RECOVERY_20261028_R2.json").open("x",encoding="utf-8") as f:
    json.dump(report,f,ensure_ascii=False,indent=2);f.write("\n")
print(json.dumps(report,ensure_ascii=False))
raise SystemExit(0 if report["cache_saved"] else 1)

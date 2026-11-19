"""Fixed Stage09 HTTP probe; no UI, terminal writes, or SSH bootstrap.

GETs use the current installed Inspire SDK transport and raw official Jupyter
access URL. This script never creates a terminal or executes a command. No URL, credentials,
raw response body, exception text, or library diagnostics are persisted.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import io
import json
import logging
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent
ACCOUNT = "sii-shared"
NOTEBOOK = "nancy-stage09-prep-20261027-r1"
WORKSPACE = "CPU资源空间"
SAFE_KEYS = {"connections", "kernels", "last_activity", "started", "name", "message", "version"}


def response_shape(response):
    try:
        raw = response.raw.read(65537, decode_content=True)
        media = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        out = {"http_status": int(response.status_code),
               "content_type": media if re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", media) else "other",
               "body_bytes_read": len(raw), "truncated": len(raw) > 65536,
               "redirect_count": len(response.history),
               "gateway_econnrefused": b"ECONNREFUSED" in raw}
        if len(raw) <= 65536:
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeError):
                out["json_type"] = None
            else:
                out["json_type"] = type(data).__name__
                if isinstance(data, list):
                    out["item_count"] = len(data)
                    keys = {k for row in data if isinstance(row, dict) for k in row}
                elif isinstance(data, dict):
                    keys = set(data)
                else:
                    keys = set()
                out["known_json_keys"] = sorted(keys & SAFE_KEYS)
                out["other_json_key_count"] = len(keys - SAFE_KEYS)
        return out
    finally:
        response.close()


def probe(report):
    import inspire
    from inspire import InspireClient
    from inspire.platform.web.browser_api.jupyter_terminal import _notebook_jupyter_url
    from inspire.platform.web.jupyter_urls import jupyter_server_base

    report["inspire_version"] = inspire.__version__
    # Fake nonsecret URL only: this never contacts a server.
    fake_base = "https://example.invalid/jupyter/notebook/fake-token/"
    canonical = jupyter_server_base(fake_base + "lab")
    after_navigation = jupyter_server_base(fake_base + "lab/terminals/1?reset")
    report["offline_url_test"] = {
        "original_lab_base_correct": canonical == fake_base,
        "terminal_page_base_correct": after_navigation == fake_base,
        "terminal_page_base_changes": canonical != after_navigation,
        "network_requests": 0,
    }
    with InspireClient(account=ACCOUNT, allow_browser=False, timeout=12,
                       operation_timeout=45, catalog_disk_cache=False) as client:
        report["stage"] = "live_notebook_resolution"
        notebook = client.notebooks.get(NOTEBOOK, workspace=WORKSPACE)
        report["notebook_status"] = notebook.status
        if notebook.status.upper() != "RUNNING":
            report["status"] = "not_running"
            return
        report["stage"] = "official_jupyter_access_url"
        with client._transport.scope(timeout=55):
            session = client._transport.session
            lab_url = _notebook_jupyter_url(session, notebook.ref.key)
            report["access_url_present"] = bool(lab_url)
            if not lab_url:
                report["status"] = "empty_access_url"
                return
            base = jupyter_server_base(lab_url)
            with client._transport.application_connection(lab_url) as http:
                for label, url, redirect in (
                    ("entrance", lab_url, True),
                    ("api_status", base + "api/status", False),
                    ("api_terminals", base + "api/terminals", False),
                ):
                    report["stage"] = "get_" + label
                    try:
                        response = http.get(url, timeout=(5, 12),
                                            allow_redirects=redirect, stream=True)
                        report["gets"][label] = response_shape(response)
                    except Exception as error:
                        report["gets"][label] = {"exception_type": type(error).__name__}
                report["xsrf_cookie_present"] = any(c.name == "_xsrf" for c in http.cookies)
        gets = report["gets"]
        healthy = all(gets.get(k, {}).get("http_status") == 200 for k in
                      ("entrance", "api_status", "api_terminals"))
        healthy = healthy and gets["api_status"].get("json_type") == "dict"
        healthy = healthy and gets["api_terminals"].get("json_type") == "list"
        report["status"] = "passed" if healthy else "get_probe_failed"


def main():
    started = time.monotonic()
    stamp = dt.datetime.now(dt.timezone.utc)
    report = {"schema": "stage09_jupyter_read_probe_v1", "status": "in_progress",
              "started_at_utc": stamp.isoformat(), "account": ACCOUNT,
              "notebook": NOTEBOOK, "workspace": WORKSPACE, "gets": {},
              "scope": "GET backend probe only; no terminal creation, remote exec, SSH bootstrap or UI",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    logging.disable(logging.CRITICAL)
    # Do not print third-party diagnostics or exception messages.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            probe(report)
        except Exception as error:
            report["status"] = "failed"
            report["exception_type"] = type(error).__name__
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    output = ROOT / ("JUPYTER_GET_DIAGNOSTIC_" + stamp.strftime("%Y%m%dT%H%M%SZ") + ".json")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"report": str(output), **report}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

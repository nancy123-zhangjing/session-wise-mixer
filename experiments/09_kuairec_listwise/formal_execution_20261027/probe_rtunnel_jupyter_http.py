"""One GET of the existing Stage09 rtunnel via the official Jupyter route.

No terminal, SSH, bootstrap, cache registration, browser, or service mutation.
URL/token, raw body and exception messages never enter the output.
"""
import contextlib
import datetime as dt
import hashlib
import io
import json
import logging
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent


def probe(report):
    from inspire import InspireClient
    from inspire.platform.web.browser_api.jupyter_terminal import _notebook_jupyter_url
    from inspire.platform.web.browser_api.playwright_notebooks import build_jupyter_proxy_url
    from inspire.platform.web.browser_api.rtunnel import _is_rtunnel_proxy_ready
    from inspire.platform.web.jupyter_urls import jupyter_server_base
    from inspire.platform.web.session import build_requests_session
    with InspireClient(account='sii-shared', allow_browser=False, timeout=12,
                       operation_timeout=45, catalog_disk_cache=False) as client:
        report['stage'] = 'live_notebook_resolution'
        notebook = client.notebooks.get('nancy-stage09-prep-20261027-r1', workspace='CPU资源空间')
        report['notebook_status'] = notebook.status
        if notebook.status.upper() != 'RUNNING':
            report['status'] = 'not_running'
            return
        with client._transport.scope(timeout=30):
            report['stage'] = 'official_jupyter_access_url'
            lab_url = _notebook_jupyter_url(client._transport.session, notebook.ref.key)
            if not lab_url:
                report['status'] = 'empty_access_url'
                return
            proxy_url = build_jupyter_proxy_url(lab_url, port=31337)
            canonical_url = jupyter_server_base(lab_url) + 'proxy/31337/'
            report['canonical_proxy_path_matches_official_builder'] = urlsplit(proxy_url).path == urlsplit(canonical_url).path
            if not report['canonical_proxy_path_matches_official_builder']:
                report['status'] = 'route_mismatch'
                return
            report['stage'] = 'get_existing_rtunnel'
            # The SDK application facade raises on non-2xx statuses. A health
            # diagnostic needs the actual response (rtunnel GET may be 404).
            # Reuse the package's official account/proxy-configured HTTP jar.
            with build_requests_session(client._transport.session, proxy_url) as http:
                response = http.get(proxy_url, timeout=(5, 12), allow_redirects=False, stream=True)
                try:
                    raw = response.raw.read(65537, decode_content=True)
                    body = raw.decode('utf-8', errors='replace')
                    media = response.headers.get('Content-Type', '').split(';', 1)[0].strip().lower()
                    report['http_status'] = int(response.status_code)
                    report['content_type'] = media if re.fullmatch(r'[a-z0-9.+-]+/[a-z0-9.+-]+', media) else 'other'
                    report['body_bytes_read'] = len(raw)
                    report['truncated'] = len(raw) > 65536
                    report['gateway_econnrefused'] = 'ECONNREFUSED' in body
                    report['html_response'] = '<html' in body.lower() or '<!doctype html' in body.lower()
                    report['plain_text_404'] = response.status_code == 404 and 'page not found' in body.lower() and not report['html_response']
                    report['official_http_readiness_predicate'] = _is_rtunnel_proxy_ready(status=response.status_code, body=body)
                    try:
                        data = json.loads(raw)
                    except (ValueError, UnicodeError):
                        report['json_type'] = None
                    else:
                        report['json_type'] = type(data).__name__
                        keys = set(data) if isinstance(data, dict) else set()
                        known = {'status', 'message', 'error', 'version', 'ready'}
                        report['known_json_keys'] = sorted(keys & known)
                        report['other_json_key_count'] = len(keys - known)
                    report['status'] = 'http_probe_complete'
                finally:
                    response.close()


def main():
    stamp = dt.datetime.now(dt.timezone.utc)
    start = time.monotonic()
    report = {'schema': 'stage09_existing_rtunnel_http_v1', 'started_at_utc': stamp.isoformat(),
              'status': 'in_progress', 'scope': 'single read-only GET; no service or cache changes',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            probe(report)
        except Exception as error:
            report['status'] = 'failed'
            report['exception_type'] = type(error).__name__
            status = re.match(r'^HTTP ([1-5][0-9]{2})(?:\D|$)', str(error))
            if status:
                report['exception_http_status'] = int(status.group(1))
    report['elapsed_seconds'] = round(time.monotonic() - start, 3)
    output = ROOT / ('RTUNNEL_HTTP_DIAGNOSTIC_' + stamp.strftime('%Y%m%dT%H%M%SZ') + '.json')
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps({'report': str(output), **report}, ensure_ascii=False))
    return 0 if report['status'] == 'http_probe_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Read-only CLI compatibility probe; print structure, never cookies or secrets."""
import json
import os
from click.testing import CliRunner
from inspire.cli.main import main as cli
from inspire.platform.web.browser_api import workspaces
from inspire.platform.web.session.auth import BOOTSTRAP_ROUTES_BODY, USER_ROUTES_PATH, _workspace_routes_from_payload

os.environ['INSPIRE_PLAYWRIGHT_CHROMIUM_CHANNEL'] = 'chrome'
unwrap = workspaces._v2_result
seen = []


def inspect(payload):
    data = unwrap(payload)
    routes = data.get('routes') or []
    seen.append(dict(keys=list(data), route_count=len(routes), routes=[
        dict(name=r.get('name'), keys=list(r), child_count=len(r.get('routes') or []),
             children=[{'name':x.get('name'), 'keys':list(x)} for x in (r.get('routes') or []) if isinstance(x,dict)])
        for r in routes if isinstance(r,dict)]))
    return data


workspaces._v2_result = inspect
lookup = workspaces._workspace_route_entries
def check_session(session, **kwargs):
    seen.append({'session_type':type(session).__name__, 'has_workspace':bool(getattr(session,'workspace_id',None)),
                 'has_cached_names':bool(getattr(session,'all_workspace_names',None))})
    try:
        return lookup(session, **kwargs)
    except workspaces.WorkspaceCapabilityError as exc:
        seen.append({'error_type':type(exc).__name__, 'error':str(exc)})
        payload = workspaces._request_json(session, 'POST', '/api/v2/user?Action=ListAccessibleWorkspaces',
            referer=workspaces._get_base_url() + '/jobs/spacesOverview', body={}, timeout=15)
        data = unwrap(payload)
        rows = data.get('items') or []
        seen.append({'live_keys':list(data), 'live_bootstrap_names':[r.get('name') for r in rows],
                     'item_keys':[list(r) for r in rows]})
        return {r['workspace_id']:{'id':r['workspace_id'],'name':r['name'],
                                  'is_fair_workspace':r.get('is_fair_workspace') is True} for r in rows}
workspaces._workspace_route_entries = check_session
result = CliRunner().invoke(cli, ['--json','--account','sii-shared','cache','refresh','--resource','workspace'])
print(json.dumps({'exit_code':result.exit_code, 'route_structure':seen, 'cli_output':result.output}, ensure_ascii=False))

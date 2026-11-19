"""Run the real CLI with the console's verified workspace-enumeration Action.

No installed CLI source is edited. All workload commands remain the original
name-only CLI, including its permission, quota and write checks. The sole shim
replaces obsolete GetRoutes workspace extraction with ListAccessibleWorkspaces,
observed in the live console index.DRzQPNV7.js and verified read-only on 2026-10-31.
"""
import os
import re
from inspire.cli.main import cli
from inspire.platform.web.browser_api import workspaces
from inspire.platform.web.session.models import DEFAULT_WORKSPACE_ID

os.environ.setdefault('INSPIRE_PLAYWRIGHT_CHROMIUM_CHANNEL', 'chrome')


def workspace_entries(session, *, base_url=None, workspace_id=None):
    base = base_url or workspaces._get_base_url()
    payload = workspaces._request_json(session, 'POST', '/api/v2/user?Action=ListAccessibleWorkspaces',
        referer=base.rstrip('/') + '/jobs/spacesOverview', body={}, timeout=15)
    data = workspaces._v2_result(payload)
    if not isinstance(data.get('items'), list):
        raise workspaces.WorkspaceEnumerationError('Live workspace schema changed again')
    result = {}
    for row in data['items']:
        wid, name = row.get('workspace_id'), row.get('name')
        if not isinstance(wid,str) or not workspaces._WS_ID_RE.match(wid) or not isinstance(name,str):
            raise workspaces.WorkspaceEnumerationError('Invalid live workspace identity')
        result[wid] = dict(id=wid, name=name, is_fair_workspace=row.get('is_fair_workspace') is True)
    session.all_workspace_ids = list(result)
    session.all_workspace_names = {k:v['name'] for k,v in result.items()}
    session.all_workspace_fair_scheduling = {k:v['is_fair_workspace'] for k,v in result.items()}
    if result and (not getattr(session,'workspace_id',None) or session.workspace_id == DEFAULT_WORKSPACE_ID):
        cpu = next((k for k,v in result.items() if v['name']=='CPU资源空间'), None)
        session.workspace_id = cpu or next(iter(result))
    if getattr(session,'account',None) not in (None,'sii-shared'):
        raise RuntimeError('Unexpected account for this experiment')
    session.save(account='sii-shared')
    return result


workspaces._workspace_route_entries = workspace_entries
if __name__ == '__main__':
    cli()

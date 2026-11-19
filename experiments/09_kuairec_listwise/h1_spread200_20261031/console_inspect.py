"""Read public console asset names to diagnose the workspace API migration."""
import json
import os
from pathlib import Path
import re
from urllib.parse import urljoin
import requests
from inspire.platform.web.session import get_web_session, build_requests_session

base = 'https://<institution-host>/'
os.environ['INSPIRE_PLAYWRIGHT_CHROMIUM_CHANNEL'] = 'chrome'
session = get_web_session(account='sii-shared')
http = build_requests_session(session, base.rstrip('/'))
response = http.get(base, timeout=25, allow_redirects=False)
if response.status_code in (301,302,303,307,308,401):
    raise RuntimeError('Authenticated console HTML did not load; no redirect/token is exposed')
response.raise_for_status()
scripts = re.findall(r'<script[^>]+src=[\"\']([^\"\']+)', response.text)
here = Path(__file__).resolve().parent
(here / 'console_public_assets.json').write_text(json.dumps({'url':response.url, 'status':response.status_code,
    'scripts':[urljoin(base,s) for s in scripts]},indent=2),encoding='utf-8')
print(json.dumps({'status':response.status_code,'scripts':scripts}))
for i, name in enumerate(scripts[:3]):
    url = urljoin(base,name)
    if not url.startswith(base):
        continue
    asset = http.get(url,timeout=40,allow_redirects=False)
    asset.raise_for_status()
    text = asset.text
    (here / ('console_asset_' + str(i) + '.js')).write_text(text,encoding='utf-8')
    hits = [text[max(0,m.start()-100):m.end()+180] for m in re.finditer(
        r'ListWorkspaces|ListUserWorkspaces|GetWorkspaceList|userWorkspaceList|spacesOverview',text)]
    print(json.dumps({'asset':name,'bytes':len(asset.content),'hits':hits[:14]},ensure_ascii=False))

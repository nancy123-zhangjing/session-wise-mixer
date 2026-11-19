"""Temporary CPU Jupyter HTTP/WebSocket fallback when CLI Chromium resets.

Uses the CLI's account session, named-notebook resolution and URL helpers.
No credentials are written or printed. No installed CLI files are changed.
"""
import argparse
import base64
import hashlib
from http.cookies import SimpleCookie
import json
import os
from pathlib import Path
import re
import shlex
import ssl
import sys
import time
from urllib.parse import urlsplit
import uuid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'.runtime/l5o_transport_deps'))
import requests
import websocket
from inspire.platform.web.browser_api import rtunnel,playwright_notebooks
from inspire.platform.web.session import build_requests_session
from inspire.cli.main import main as inspire


def cookies_for(http,url):
    return http.prepare_request(requests.Request('GET',url)).headers.get('Cookie','')


def execute(http,lab_url,command,timeout,blob=None):
    base=rtunnel._jupyter_server_base(lab_url)
    jar=SimpleCookie();jar.load(cookies_for(http,base))
    headers={'Referer':lab_url}
    if '_xsrf' in jar:headers['X-XSRFToken']=jar['_xsrf'].value
    created=http.post(base+'api/terminals',headers=headers,timeout=20)
    created.raise_for_status()
    name=created.json()['name']
    ws=None
    try:
        url=rtunnel._build_terminal_websocket_url(lab_url,name)
        ws=websocket.create_connection(url,timeout=20,cookie=cookies_for(http,base),
            origin=f'{urlsplit(base).scheme}://{urlsplit(base).netloc}',
            sslopt={'cert_reqs':ssl.CERT_NONE},http_no_proxy=[urlsplit(base).hostname])
        deadline=time.monotonic()+timeout
        marker='L5O_DONE_'+uuid.uuid4().hex
        sent=False;ready=False;blob_sent=False;out=''
        ready_marker='L5O_READY_'+uuid.uuid4().hex
        while time.monotonic()<deadline:
            ws.settimeout(min(20,max(1,deadline-time.monotonic())))
            try:data=json.loads(ws.recv())
            except websocket.WebSocketTimeoutException:continue
            if data[0]=='stdout':
                if not sent:
                    ws.send(json.dumps(['stdin','stty -echo; PS1=""; PS2=""; printf "\\n'+ready_marker+'\\n"\r']))
                    sent=True
                else:
                    out+=data[1]
                    if not ready and re.search(r'\n'+ready_marker+r'\r?\n',out):
                        ready=True;out=''
                        suffix='status=$?; '+('stty icanon; ' if blob is not None else '')+'printf "\\n'+marker+':%s\\n" "$status"\r'
                        inner=command.rstrip()+('; ' if blob is not None else '\n')+suffix
                        for start in range(0,len(inner),65536):
                            ws.send(json.dumps(['stdin',inner[start:start+65536]]))
                    if blob is not None and ready and not blob_sent and 'L5O_BLOB_READY' in out:
                        blob_sent=True
                        for start in range(0,len(blob),16384):
                            ws.send(json.dumps(['stdin',blob[start:start+16384]]))
                    match=re.search(r'\n'+marker+r':(\d+)\r?\n',out)
                    if match:
                        clean=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',out[:match.start()]).replace('\r','').strip()
                        return dict(returncode=int(match[1]),output=clean)
        raise TimeoutError('CPU Jupyter command timed out')
    finally:
        if ws is not None:ws.close()
        http.delete(base+'api/terminals/'+str(name),headers=headers,timeout=10)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--notebook',default='nancy-s09-e-prep-1101-r1')
    p.add_argument('--command',default='pwd')
    p.add_argument('--command-file',type=Path)
    p.add_argument('--blob-file',type=Path,help='Base64 body for a command with the L5O_BLOB_READY handshake')
    p.add_argument('--timeout',type=int,default=30)
    p.add_argument('--upload',type=Path)
    p.add_argument('--remote-file')
    p.add_argument('--sync-source',type=Path)
    p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.command_file:a.command=a.command_file.read_text(encoding='utf-8-sig')
    os.environ.update(NO_PROXY='.sii.edu.cn',HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='')
    rtunnel.probe_existing_rtunnel_proxy_url=lambda **kw:None
    original=playwright_notebooks._resolve_direct_lab_url
    def action(url,**kw):
        lab=original(url,**kw)
        http=build_requests_session(kw['session'],lab)
        try:
            http.get(lab,timeout=15).raise_for_status()
            if a.sync_source:
                remote='/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09/l5o_stage0_20261117_v1/source_dataset'
                files=sorted((a.sync_source/'events').glob('*.parquet'))
                files += [a.sync_source/n for n in ('manifest.json','item_map.json','user_map.json','item_text.jsonl',
                    'text_repairs.jsonl','item_catalog.parquet','user_catalog.parquet')]
                rows=[]
                for file in files:
                    name=file.relative_to(a.sync_source).as_posix()
                    target=remote+'/'+name
                    digest=hashlib.sha256(file.read_bytes()).hexdigest()
                    probe=execute(http,lab,'if test -f '+shlex.quote(target)+'; then sha256sum '+shlex.quote(target)+'; fi',30)
                    if probe['returncode']:raise RuntimeError(probe)
                    if probe['output']:
                        if probe['output'].split()[0]!=digest:raise RuntimeError('Existing source differs: '+name)
                        row=dict(path=name,bytes=file.stat().st_size,sha256=digest,reused=True)
                    else:
                        body=base64.b64encode(file.read_bytes()).decode()
                        script=("import sys,base64,hashlib,pathlib; print('L5O_BLOB_READY',flush=True); "
                            +f"b=base64.b64decode(sys.stdin.buffer.read({len(body)})); "
                            +f"assert hashlib.sha256(b).hexdigest()=={digest!r}; p=pathlib.Path({target!r}); "
                            +"p.parent.mkdir(parents=True,exist_ok=True); assert not p.exists(); "
                            +"tmp=p.with_suffix(p.suffix+'.part'); tmp.write_bytes(b); tmp.replace(p); print('SHA_VERIFIED')")
                        cmd='stty -icanon -echo; python -c '+shlex.quote(script)
                        received=execute(http,lab,cmd,max(a.timeout,180),blob=body)
                        if received['returncode'] or 'SHA_VERIFIED' not in received['output']:raise RuntimeError(received)
                        row=dict(path=name,bytes=file.stat().st_size,sha256=digest,reused=False)
                    rows.append(row)
                    print(json.dumps(dict(file=name,verified=len(rows),total=len(files)),ensure_ascii=False),flush=True)
                    if a.receipt:a.receipt.write_text(json.dumps(dict(remote=remote,files=rows,complete=len(rows)==len(files)),indent=2),encoding='utf-8')
                result=dict(remote=remote,files=rows,complete=True)
            elif a.upload:
                if not a.remote_file or not a.remote_file.startswith('nancy/') or '..' in Path(a.remote_file).parts:
                    raise ValueError('Upload must be an explicit nancy/ relative file')
                base=rtunnel._jupyter_server_base(lab)
                jar=SimpleCookie();jar.load(cookies_for(http,base))
                headers={'Referer':lab}
                if '_xsrf' in jar:headers['X-XSRFToken']=jar['_xsrf'].value
                target=base+'api/contents/'+a.remote_file
                exists=http.get(target+'?content=0',timeout=15)
                if exists.status_code!=404:raise FileExistsError('Refusing to overwrite upload path')
                r=http.put(target,headers=headers,json=dict(type='file',format='base64',
                    content=base64.b64encode(a.upload.read_bytes()).decode()),timeout=60)
                r.raise_for_status()
                result=dict(uploaded=a.remote_file,bytes=a.upload.stat().st_size)
            else:
                blob = a.blob_file.read_text(encoding='ascii') if a.blob_file else None
                result=execute(http,lab,a.command,a.timeout,blob=blob)
            if a.receipt:a.receipt.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(result,ensure_ascii=False),flush=True)
        finally:http.close()
        raise SystemExit(0)
    playwright_notebooks._resolve_direct_lab_url=action
    old=playwright_notebooks.open_notebook_lab
    def direct(*args,**kw):
        kw['prefer_direct']=True
        return old(*args,**kw)
    playwright_notebooks.open_notebook_lab=direct
    inspire.main(args=['--account','sii-shared','--json','notebook','connection','refresh',a.notebook,
        '--workspace','CPU资源空间','--timeout','20'],standalone_mode=False)


if __name__=='__main__':main()

import hashlib, json, pathlib, tarfile, sys

base = pathlib.Path('/inspire/hdd/global_user/<inspire-account>/nancy/experiment1_stage09')
source = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
rows = []
def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8*1024*1024), b''):
        h.update(block)
    return h.hexdigest()
for row in source['files']:
    name = pathlib.PurePosixPath(row['relative_path']).name
    if '.partial' in name:
        continue
    if name.startswith('stage09_bge'):
        path = base / 'goal_20261027_r1/delivery' / name
    elif 'spread200' in name:
        path = base / 'h1_spread200_20261031_r1/delivery' / name
    elif 'block40' in name:
        path = base / 'h1_block40_r6_single_20261101_r1/delivery' / name
    else:
        path = base / name
    item = dict(row, remote=str(path))
    with path.open('rb') as f:
        item['remote_sha256'] = digest(f)
    item['remote_bytes'] = path.stat().st_size
    if item['remote_bytes'] != row['bytes'] or item['remote_sha256'] != row['sha256']:
        raise ValueError('Mismatch: '+str(path))
    item['remote_verified'] = True
    if '.part' not in name:
        members = []
        with tarfile.open(path, 'r|*') as tar:
            for member in tar:
                if member.isfile():
                    with tar.extractfile(member) as f:
                        members.append(dict(name=member.name, bytes=member.size, sha256=digest(f)))
        if not members:
            raise ValueError('Empty archive')
        item['archive_members'] = members
        item['restore_verified'] = True
        item['restore_scope'] = 'All regular members streamed from archive and hashed; no model execution'
    rows.append(item)
    print(json.dumps(dict(verified=name,bytes=row['bytes'])),flush=True)
pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(files=rows,status='remote_bytes_and_archive_read_passed')),encoding='utf-8')

#!/usr/bin/env python3
"""Fetch the inputs of the 27 evaluated modes at their pinned revisions.

    python scripts/fetch_public_sources.py list
    python scripts/fetch_public_sources.py all [--data-root DIR]
    python scripts/fetch_public_sources.py NAME [--data-root DIR]

public_sources.json declares every source, its revision and the local file
layout the task loaders read. Files already present are kept. Gated sources
download only with your own provider credentials; otherwise their steps are
printed. Afterwards run `python -m s1cogbench doctor`.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / 'public_sources.json').read_text())


class Gated(Exception):
    pass


def run(command, **kwargs):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(command, check=True, **kwargs)


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    print('+ download', url, flush=True)
    partial = path.with_name(path.name + '.part')
    with urllib.request.urlopen(url, timeout=600) as response, partial.open('wb') as stream:
        shutil.copyfileobj(response, stream, 1 << 20)
    partial.replace(path)


def github(name, source, root):
    """The repository tree at the pinned commit, from GitHub's archive (no git needed)."""
    dest = root / source['destination']
    head = dest / '.git/HEAD'
    if head.exists() and head.read_text().strip() != source['revision']:
        raise SystemExit(f'{dest} is a checkout at {head.read_text().strip()}, not the evaluated '
                         f'{source["revision"]}; remove it and fetch again.')
    if dest.exists():
        return
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        path = Path(tmp) / 'source.tar.gz'
        download(f"https://codeload.github.com/{source['repository']}/tar.gz/{source['revision']}", path)
        with tarfile.open(path) as t:
            t.extractall(Path(tmp) / 'tree', filter='data')
        top, = (Path(tmp) / 'tree').iterdir()
        dest.parent.mkdir(parents=True, exist_ok=True)
        top.rename(dest)


def files(name, source, root):
    for local, url in source['files'].items():
        if not (root / local).exists():
            download(url, root / local)
    for local in source.get('empty', []):
        (root / local).touch()
    if source.get('derive') == 'hybrid_folds':
        hybrid_folds(root)


def hybrid_folds(root):
    """Held-out folds of the Hybrid paper: MLUtils kfolds(n, 8) per subject, contiguous
    validation blocks, the first n mod 8 folds one row larger (the published code's split;
    no fold file is published)."""
    target = root / 'data/hybrid_paper_matched/fold_assignments.csv'
    if target.exists():
        return
    rows = ['subject,author_X_row_1based,validation_fold_1based']
    for subject in range(1, 21):
        with (root / f'external/HybridModellingProject/Data/fit/data/{subject}/X.csv').open() as f:
            n = sum(1 for _ in f) - 1
        size, extra = divmod(n, 8)
        row = 0
        for fold in range(1, 9):
            for _ in range(size + (fold <= extra)):
                row += 1
                rows.append(f'{subject},{row},{fold}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(rows) + '\n')


def archive(name, source, root):
    """Download an archive once, keep only the listed members, then drop the archive."""
    wanted = {local: member for local, member in source['files'].items() if not (root / local).exists()}
    if wanted:
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            path = Path(tmp) / source['archive_name']
            download(source['url'], path)
            with zipfile.ZipFile(path) as z:
                for local, member in wanted.items():
                    (root / local).parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, (root / local).open('wb') as dst:
                        shutil.copyfileobj(src, dst)
    for local, tracked in source.get('copy', {}).items():
        if not (root / local).exists():
            (root / local).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / tracked, root / local)


def julia(name, source, root):
    tools = root / '.tools'
    binary = tools / 'julia-1.4.1/bin/julia'
    if not binary.exists():
        tools.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tools) as tmp:
            path = Path(tmp) / 'julia.tar.gz'
            download(source['url'], path)
            with tarfile.open(path) as t:
                t.extractall(tools, filter='data')
    env = tools / 'mouselab-julia141-env'
    env.mkdir(exist_ok=True)
    for file in ('Project.toml', 'Manifest.toml'):
        shutil.copyfile(ROOT / source['environment'] / file, env / file)
    depot = tools / 'julia-depot-1.4.1'
    depot.mkdir(exist_ok=True)
    run([str(binary), '--startup-file=no', f'--project={env}', '-e', 'using Pkg; Pkg.instantiate()'],
        env={**os.environ, 'JULIA_DEPOT_PATH': str(depot), 'JULIA_PROJECT': str(env)})


def huggingface(name, source, root):
    target = root / source['destination']
    if target.exists():
        return
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(source['repo_id'], source['filename'], repo_type='dataset',
                               revision=source['revision'])
    except Exception as exc:  # noqa: BLE001 - no access yet: report and print the steps
        raise Gated(f'{name}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}') from None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)


KINDS = {'github': github, 'files': files, 'archive': archive, 'julia': julia, 'huggingface': huggingface}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', choices=['list', 'all', *SOURCES])
    p.add_argument('--data-root', type=Path, default=ROOT)
    args = p.parse_args()
    root = args.data_root.resolve()
    if args.source == 'list':
        for name, s in SOURCES.items():
            print(f"{name:20s} {s['access']:7s} {s['license'][:40]:40s} {', '.join(s['modes'])}")
        return
    names = list(SOURCES) if args.source == 'all' else [args.source]
    fetched, gated = [], []
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        try:
            KINDS[SOURCES[name]['kind']](name, SOURCES[name], root)
            fetched.append(name)
        except Gated as exc:
            print(exc, file=sys.stderr)
            gated.append(name)
    print(json.dumps({'fetched': fetched, 'not_fetched': gated, 'data_root': str(root)}))
    for name in gated:
        print(f'{name} needs your access first:')
        for i, step in enumerate(SOURCES[name]['steps'], 1):
            print(f'  {i}. {step}')
    print('Next: python -m s1cogbench doctor' + ('' if root == ROOT else f' --data-root {root}'))
    raise SystemExit(1 if gated else 0)


if __name__ == '__main__':
    main()

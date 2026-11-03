#!/usr/bin/env python3
"""Prepare homepage/ without model calls: python scripts/build_homepage.py [--package]

Copies the aggregate files to homepage/data and writes homepage/docs from docs/ (Markdown and
HTML; needs the markdown package). With --package it also writes the downloads: the paper PDF
from paper/ and the source archive. Without it the downloads stay as they are.
"""
import argparse
from pathlib import Path
import re
import shutil
import zipfile

import markdown

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'homepage'
GITHUB = 'https://github.com/dengzhe-hou/S1CogBench/blob/main/'
# Fetched or private inputs, and the outputs of runs, stay out of the source archive.
NOT_ARCHIVED = {'.git', '.runtime', 'runs', 'models', 'data', 'external', 'literature', '.tools'}
PAGE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · S1CogBench</title>
<link rel="icon" href="../assets/mark.svg" type="image/svg+xml">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Figtree:wght@600&family=Noto+Sans:wght@400;600;700&display=swap">
<link rel="stylesheet" href="../style.css"></head><body>
<nav class="navbar" aria-label="Main navigation"><a href="../index.html#leaderboard">Leaderboard</a><a href="../index.html#tasks">Tasks</a><a href="../index.html#evaluate">Evaluate</a><a href="../index.html#resources">Resources</a></nav>
<main class="doc-page"><p class="doc-back"><a href="../index.html#resources">Back to S1CogBench</a></p>
{body}
</main></body></html>
'''


def site_link(match):
    """Point a link of a docs page at its place on the site, or at the repository."""
    link = match.group(1)
    if re.match(r'[a-z]+:|#', link):
        return match.group(0)
    path, _, anchor = link.partition('#')
    anchor = '#' + anchor if anchor else ''
    if '/' not in path and path.endswith('.md'):
        target = path[:-3] + '.html'
    elif path.startswith('../results/aggregate/'):
        target = '../data/' + path.split('/')[-1]
    elif path == '../public_sources.json':
        target = '../data/public_sources.json'
    elif path == '../LICENSE':
        target = '../LICENSE.txt'
    else:
        target = GITHUB + str(Path('docs', path).as_posix()).replace('docs/../', '')
    return f'href="{target}{anchor}"'


def write_docs():
    out = SITE / 'docs'
    out.mkdir(exist_ok=True)
    for old in out.glob('*'):
        if old.is_file():
            old.unlink()
    for source in sorted((ROOT / 'docs').glob('*.md')):
        text = source.read_text()
        shutil.copy2(source, out / source.name)
        title = text.splitlines()[0].lstrip('# ').strip()
        body = markdown.markdown(text, extensions=['tables', 'fenced_code'])
        body = re.sub(r'href="([^"]+)"', site_link, body)
        (out / (source.stem + '.html')).write_text(PAGE.format(title=title, body=body))


def package():
    (SITE / 'downloads').mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'paper/s1cogbench.pdf', SITE / 'downloads/s1cogbench-paper.pdf')
    with zipfile.ZipFile(SITE / 'downloads/s1cogbench-source.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(ROOT.rglob('*')):
            rel = file.relative_to(ROOT)
            if not file.is_file() or file.is_symlink() or rel.parts[0] in NOT_ARCHIVED:
                continue
            if '.venv' in rel.parts or '__pycache__' in rel.parts or file.name == '.env':
                continue
            if rel.parts[:2] in {('homepage', 'downloads'), ('homepage', 'data')}:
                continue
            archive.write(file, rel.as_posix())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', action='store_true', help='Also write the paper PDF and source archive to homepage/downloads.')
    args = parser.parse_args()
    shutil.copytree(ROOT / 'results/aggregate', SITE / 'data', dirs_exist_ok=True)
    shutil.copy2(ROOT / 'public_sources.json', SITE / 'data/public_sources.json')
    shutil.copy2(ROOT / 'LICENSE', SITE / 'LICENSE.txt')
    write_docs()
    if args.package:
        package()
    print('Static homepage prepared at', SITE, '(downloads rewritten)' if args.package else '(downloads unchanged)')


if __name__ == '__main__':
    main()

"""Check that a CLM embedding server honours truncation_side before the left-truncation control.

Two texts that share their last 2,048+ tokens but differ at the start must embed identically with
truncation_side=left, and differently under the official default (the first 2,048 tokens).

Usage: python scripts/check_clm_truncation_side.py http://127.0.0.1:8090/v1/embeddings
"""
import base64
import sys

import numpy as np
import requests

url = sys.argv[1]
tail = 'gamma delta ' * 1500
texts = ['alpha ' * 3000 + tail, 'beta ' * 3000 + tail]


def cosine(extra):
    body = {'model': 'qwen3-8b', 'input': texts, 'encoding_format': 'base64', 'truncate_prompt_tokens': 2048, **extra}
    r = requests.post(url, json=body, timeout=300)
    r.raise_for_status()
    v = [np.frombuffer(base64.b64decode(d['embedding']), dtype=np.float32) for d in r.json()['data']]
    return float(v[0] @ v[1] / np.linalg.norm(v[0]) / np.linalg.norm(v[1]))


left, official = cosine({'truncation_side': 'left'}), cosine({})
print(f'cosine with truncation_side=left {left:.6f}; with the official default {official:.6f}')
if left < 0.9999 or official > 0.999:
    sys.exit('truncation side is not what the control needs')

#!/usr/bin/env python3
"""
Convert PCA W-direction .pt files into the small JSON the web app loads.

Input:  a torch.save dict with a `components` matrix [dim, K] (columns are the
        principal directions in W space), plus eigenvalues / explained variance.
Output: <name>.json next to each <name>.pt in models/w_directions/, keeping the
        first `count` directions (default 2) as plain float arrays.

Usage:  python3 scripts/w-directions.py [count]
"""
import glob
import json
import os
import sys

import torch

COUNT = int(sys.argv[1]) if len(sys.argv) > 1 else 2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'models', 'w_directions')


def main():
    paths = sorted(glob.glob(os.path.join(SRC, '*.pt')))
    files = []
    for p in paths:
        obj = torch.load(p, map_location='cpu', weights_only=False)
        comp = obj['components'].float()  # [dim, K]
        k = min(COUNT, comp.shape[1])
        out = {
            'source': os.path.basename(p),
            'count': k,
            'dim': int(comp.shape[0]),
            'directions': [comp[:, i].tolist() for i in range(k)],
            'eigenvalues': obj['eigenvalues'].float()[:k].tolist(),
            'explainedVarianceRatio': obj['explained_variance_ratio'].float()[:k].tolist(),
            'truncation': float(obj.get('truncation', 1.0)),
            'numSamples': int(obj.get('num_samples', 0)),
        }
        dst = os.path.splitext(p)[0] + '.json'
        with open(dst, 'w') as f:
            json.dump(out, f)
        files.append(os.path.basename(dst))
        print(f'wrote {os.path.relpath(dst, ROOT)} ({k} directions x {comp.shape[0]})')

    # Manifest so the client only requests existing direction files (no 404s).
    manifest = {'files': sorted(files)}
    with open(os.path.join(SRC, 'manifest.json'), 'w') as f:
        json.dump(manifest, f)
    print(f'wrote {os.path.relpath(os.path.join(SRC, "manifest.json"), ROOT)} ({len(files)} file(s))')


if __name__ == '__main__':
    main()

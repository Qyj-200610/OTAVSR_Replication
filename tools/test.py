#!/usr/bin/env python
# Copyright (c) OTAVSR. All rights reserved.
"""Test a model, mirroring ``mmpretrain/tools/test.py``."""
import os.path as _osp
import sys as _sys

_root = _osp.dirname(_osp.abspath(__file__))
while not _osp.isdir(_osp.join(_root, 'otavsr')) and _osp.dirname(_root) != _root:
    _root = _osp.dirname(_root)
_sys.path.insert(0, _root)
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction

import otavsr  # noqa: F401  (populate the registries)
from otavsr.apis import test_model


def parse_args():
    parser = argparse.ArgumentParser(description='Test an OT-AVSR model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', help='checkpoint to evaluate')
    parser.add_argument('--work-dir', help='directory for logs')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config keys, e.g. '
        'test_dataloader.dataset.pipeline.1.eval_snr=0')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    work_dir = args.work_dir or osp.join(
        'work_dirs', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(work_dir, exist_ok=True)

    metrics = test_model(cfg, checkpoint=args.checkpoint, work_dir=work_dir)
    print('\nResults:')
    for name, value in metrics.items():
        print(f'  {name}: {value:.4f}')


if __name__ == '__main__':
    main()

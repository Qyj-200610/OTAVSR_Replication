#!/usr/bin/env python
# Copyright (c) OTAVSR. All rights reserved.
"""Train a model, mirroring ``mmpretrain/tools/train.py``."""
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
from otavsr.apis import train_model
from otavsr.version import __version__


def parse_args():
    parser = argparse.ArgumentParser(description='Train an OT-AVSR model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--work-dir', help='directory for logs and ckpts')
    parser.add_argument(
        '--resume',
        nargs='?',
        const='auto',
        help='resume from checkpoint (auto = latest in work_dir)')
    parser.add_argument(
        '--load-from', help='load weights before training (e.g. a baseline)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config keys, e.g. model.ot_loss.lam=0.2')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    work_dir = args.work_dir or osp.join(
        'work_dirs', osp.splitext(osp.basename(args.config))[0])
    cfg.work_dir = work_dir
    if args.seed is not None:
        cfg.randomness = dict(seed=args.seed)
    cfg.meta = dict(otavsr_version=__version__, config=args.config)

    print(f'otavsr {__version__}')
    print(f'Config:\n{cfg.pretty_text}')

    os.makedirs(work_dir, exist_ok=True)
    cfg.dump(osp.join(work_dir, osp.basename(args.config)))

    train_model(
        cfg,
        work_dir=work_dir,
        resume=args.resume == 'auto',
        load_from=args.load_from)


if __name__ == '__main__':
    main()

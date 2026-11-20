#!/usr/bin/env python
# Copyright (c) OTAVSR. All rights reserved.
"""Print the fully merged config of an experiment (``_base_`` resolved).

Usage::

    python tools/misc/print_config.py configs/table5_sota/setting3.py

Useful to check that a 5-line child config really overrides only what you
think it overrides -- mmengine merges dicts recursively, so
``model = dict(ot_loss=dict(lam=0.05))`` keeps every other ``ot_loss`` key
import os.path as _osp
import sys as _sys

_root = _osp.dirname(_osp.abspath(__file__))
while not _osp.isdir(_osp.join(_root, 'otavsr')) and _osp.dirname(_root) != _root:
    _root = _osp.dirname(_root)
_sys.path.insert(0, _root)
from the base config.
"""
import argparse
import os.path as osp

from mmengine.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Print the merged config')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--keys', nargs='+', help='only print these top-level keys')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    print(f'# merged from {args.config}')
    if args.keys:
        for key in args.keys:
            value = cfg
            for part in key.split('.'):
                value = value[part]
            print(f'{key} = {value}')
        return
    print(cfg.pretty_text)


if __name__ == '__main__':
    main()

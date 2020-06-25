#!/usr/bin/env python3

import re
import argparse

import numpy as np
import pandas
import wandb
from tabulate import tabulate

def re_group(s, rexp):
    m = rexp.match(s)
    if len(m.groups()) == 0:
        return m.group(0)
    else:
        return m.groups()[-1]

def table_wandb(path, metrics, filter_re, group_re, name_re):
    api = wandb.Api()
    groups = {}
    group_names = {}
    for run in api.runs(path):
        if not filter_re.match(run.name):
            continue

        group = re_group(run.name, group_re)
        name = re_group(run.name, name_re)

        row = [run.summary.get(metric, np.nan) for metric in metrics]
        groups.setdefault(group, []).append(row)
        group_names[group] = name

    groups_agg = {group_names[group]: np.nanmean(rows) for group, rows in groups.items()}
    df = pandas.DataFrame.from_dict(groups_agg, orient='index', columns=metrics)
    print(tabulate(df, ['run'] + metrics, showindex='always'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Display table for results',
        #formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('backend', choices=['wandb', 'dryrun'], help='Parse results from X')
    parser.add_argument('path', type=str, help='string path where results are')
    parser.add_argument('metrics', type=str, nargs='+', help='metric(s) to show in tablee')
    parser.add_argument('-f', '--filter_re', type=str, default='.*', help='show only runs that match this regex. Default: match everything')
    parser.add_argument('-g', '--group_re', type=str, default=None, help='group runs based on this regex. Default: same as filter_re')
    parser.add_argument('-n', '--name_re', type=str, default=None, help='regex to show name of run in table. Default: same as group_re')

    args = parser.parse_args()

    filter_re = re.compile(args.filter_re)
    if args.group_re is None:
        group_re = filter_re
    else:
        group_re = re.compile(args.group_re)
    if args.name_re is None:
        name_re = group_re
    else:
        name_re = re.compile(args.name_re)

    if args.backend == 'wandb':
        table_wandb(args.path, args.metrics, filter_re, group_re, name_re)
    elif args.backend == 'dryrun':
        print(tabulate([
            ['path', args.path],
            ['metrics', args.metrics],
            ['filter_re', filter_re],
            ['group_re', group_re],
            ['name_re', name_re],
        ]))
    else:
        raise NotImplementedError

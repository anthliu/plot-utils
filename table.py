#!/usr/bin/env python3

import re
import argparse
from pprint import pprint

import numpy as np
import pandas
import wandb
from tabulate import tabulate

HUMAN_ATARI_SCORES = {
    'Alien': 7127.7,
    'Amidar': 1719.5,
    'Assault': 1496,
    'Asterix': 8503.2,
    'BankHeist': 753.1,
    'BattleZone': 37800,
    'Boxing': 12.1,
    'Breakout': 31.8,
    'ChopperCommand': 9882,
    'CrazyClimber': 35411,
    'DemonAttack': 3401,
    'Freeway': 29.6,
    'Frostbite': 4334.7,
    'Gopher': 2412.5,
    'Hero': 30826.4,
    'Jamesbond': 406.7,
    'Kangaroo': 3035,
    'Krull': 2665.5,
    'KungFuMaster': 22736.3,
    'MsPacman': 15693,
    'Pong': 14.6,
    'PrivateEye': 69571.3,
    'Qbert': 13455,
    'RoadRunner': 7845,
    'Seaquest': 42054.7,
    'UpNDown': 11693.2,
}

SOTA_ATARI_SCORES = {
    'Alien': 702.5,
    'Amidar': 100.2,
    'Assault': 490.3,
    'Asterix': 577.9,
    'BankHeist': 205.3,
    'BattleZone': 6240,
    'Boxing': 5.1,
    'Breakout': 14.3,
    'ChopperCommand': 870.1,
    'CrazyClimber': 20072.2,
    'DemonAttack': 1086,
    'Freeway': 20,
    'Frostbite': 889.9,
    'Gopher': 678.0,
    'Hero': 4083.7,
    'Jamesbond': 330.3,
    'Kangaroo': 1282.6,
    'Krull': 4163,
    'KungFuMaster': 7649,
    'MsPacman': 1015.9,
    'Pong': -17.1,
    'PrivateEye': 50.4,
    'Qbert': 769.1,
    'RoadRunner': 8296.3,
    'Seaquest': 299.4,
    'UpNDown': 3134.8,
}

CURL_ATARI_SCORES = {
    'Alien': 558.2,
    'Amidar': 142.1,
    'Assault': 600.6,
    'Asterix': 734.5,
    'BankHeist': 131.6,
    'BattleZone': 14870,
    'Boxing': 1.2,
    'Breakout': 4.9,
    'ChopperCommand': 1058.5,
    'CrazyClimber': 12146.5,
    'DemonAttack': 817.6,
    'Freeway': 26.7,
    'Frostbite': 1181.3,
    'Gopher': 669.3,
    'Hero': 6279.3,
    'Jamesbond': 471.0,
    'Kangaroo': 872.5,
    'Krull': 4229.6,
    'KungFuMaster': 14307.8,
    'MsPacman': 1465.5,
    'Pong': -16.5,
    'PrivateEye': 218.4,
    'Qbert': 1042.4,
    'RoadRunner': 5661,
    'Seaquest': 384.5,
    'UpNDown': 2955.2,
}

def re_group(s, rexp):
    m = rexp.match(s)
    if len(m.groups()) == 0:
        return s
    else:
        return m.groups()

def re_group_atts(s, rexp):
    m = rexp.match(s)
    return m.groupdict()

def merge_atari_name(name):
    if not isinstance(name, str):
        for sub in name:
            maybe_atari = merge_atari_name(sub)
            if maybe_atari in HUMAN_ATARI_SCORES:
                return maybe_atari
        return name
    for game_name in HUMAN_ATARI_SCORES:
        if game_name in name:
            return game_name
    try_name = ''.join(word.capitalize() for word in name.split('_'))
    if try_name in HUMAN_ATARI_SCORES:
        return try_name
    return name

def table_wandb(path, metrics, filter_re, group_re, name_re):
    api = wandb.Api()
    groups = {}
    group_names = {}
    group_atts = {}
    all_atts = set()
    for run in api.runs(path):
        if not filter_re.match(run.name):
            continue
        if run.state != 'finished':
            continue

        group = re_group(run.name, group_re)
        name = re_group(run.name, name_re)
        atts = re_group_atts(run.name, group_re)

        if True:
            row = [run.summary.get(metric, np.nan) for metric in metrics]
        else:
            hist = run.history()
            row = [np.nanmax(hist[metric]) for metric in metrics]
        groups.setdefault(group, []).append(row)
        group_names[group] = name
        for att, val in atts.items():
            all_atts.add(att)
            group_atts.setdefault(group, {}).setdefault(att, val)

    groups_agg = {group_names[group]: [group_atts[group].get(att, None) for att in all_atts] + list(np.nanmean(rows, axis=0)) for group, rows in sorted(groups.items(), key=lambda x: x[0])}
    df = pandas.DataFrame.from_dict(groups_agg, orient='index', columns=list(all_atts) + metrics)
    try:
        print(df.groupby(['game', 'tag']).mean().to_string())
    except Exception:
        print(df.to_string())

    #print(tabulate(df, ['Run'] + metrics, showindex='always'))
    scores = dict(df[metrics[0]])
    scores = {merge_atari_name(key): value for key, value in scores.items()}
    mergewith = lambda d1, d2, f: {key: f(value, d2[key]) for key, value in d1.items()}
    normalized_scores = mergewith(scores, HUMAN_ATARI_SCORES, lambda x, y: x / y)
    sota_normalized_scores = mergewith(SOTA_ATARI_SCORES, HUMAN_ATARI_SCORES, lambda x, y: x / y)
    curl_normalized_scores = mergewith(CURL_ATARI_SCORES, HUMAN_ATARI_SCORES, lambda x, y: x / y)
    print('SOTA comparison:')
    pprint(mergewith(scores, SOTA_ATARI_SCORES, lambda x, y: x / y))
    print('SOTA comparison sorted:')
    pprint(sorted(mergewith(scores, SOTA_ATARI_SCORES, lambda x, y: x / y).items(), key=lambda x: x[1]))
    print('CURL comparison sorted:')
    pprint(sorted(mergewith(scores, CURL_ATARI_SCORES, lambda x, y: x / y).items(), key=lambda x: x[1]))
    print('SOTA comparison worst 8:')
    print([tup[0] for tup in sorted(mergewith(scores, SOTA_ATARI_SCORES, lambda x, y: x / y).items(), key=lambda x: x[1])[:8]])
    median_normalized_score = np.median(list(normalized_scores.values()))
    median_sota_normalized_score = np.median(list(sota_normalized_scores.values()))
    median_curl_normalized_score = np.median(list(curl_normalized_scores.values()))
    print(f'Median normalized score: {median_normalized_score}')
    print(f'Median SOTA normalized score: {median_sota_normalized_score}')
    print(f'Median CURL normalized score: {median_curl_normalized_score}')

    #print(f'Sorted normalized:\t{list(sorted(normalized_scores.values()))}')
    #print(f'Sorted sota normalized:\t{list(sorted(sota_normalized_scores.values()))}')

    print('Normalized: \t', *(f'{x:.2f}' for x in sorted(normalized_scores.values())))
    print('SOTA norm: \t', *(f'{x:.2f}' for x in sorted(sota_normalized_scores.values())))
    print('CURL norm: \t', *(f'{x:.2f}' for x in sorted(curl_normalized_scores.values())))

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

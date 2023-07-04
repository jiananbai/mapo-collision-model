import numpy as np
import os
from datetime import datetime
import yaml
import argparse
from pathlib import Path
import shutil

from utils.dotdic import DotDic
from configs.init_args import init_args
from runner import Runner


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('-s', '--save', action='store_true', help='save results')
    parse.add_argument('-v', '--verbose', action='store_true', help='print epoch results')
    parse.add_argument('--result_dir', type=str, default='./results')
    parse.add_argument('--env', type=str, default='collision', help='collision or o_gfra or n_gfra')
    parse.add_argument('--policy', type=str, default='mapo', help='mapo or qmix')
    parse.add_argument('--approx', type=str, default='log-sum-exp', help='log-sum-exp or virtual-queue')
    parse.add_argument('--combining', type=str, default='zf', help='zf or mr')
    parse.add_argument('--learn_power', action='store_true', help='learn power or not')
    parse.add_argument('--pre_alloc', action='store_true', help='pre-allocate pilots or not')
    parse.add_argument('--feedback', action='store_true', help='feedback or not')

    parse_args = vars(parse.parse_args())

    default_args = yaml.load(open('configs/default'), Loader=yaml.FullLoader)

    args = DotDic({**default_args, **parse_args})

    # args.verbose = True
    # args.save = True
    # # args.policy = 'qmix'
    # args.feedback = True
    # # args.pre_alloc = True

    args = init_args(args)

    if args.save:
        option_str = '_'.join([args.env, args.policy, args.approx, args.combining,
                               'LearnPower' if args.learn_power else 'NoLearnPower',
                               'PreAlloc' if args.pre_alloc else 'NoPreAlloc',
                               'Feedback' if args.feedback else 'NoFeedback'])
        # args.result_path = os.path.join(args.result_dir, option_str, datetime.now().strftime('%y%m%d%H%M'))
        args.result_path = os.path.join(args.result_dir, option_str)
        i_trial = 1
        while os.path.exists(os.path.join(args.result_path, 'trial-' + str(i_trial))):
            i_trial += 1
        args.result_path = os.path.join(args.result_path, 'trial-' + str(i_trial))

        os.makedirs(args.result_path, exist_ok=False)
        os.makedirs(os.path.join(args.result_path, 'data'), exist_ok=False)
        os.makedirs(os.path.join(args.result_path, 'model'), exist_ok=False)

        file = os.path.join(args.result_path, 'meta.txt')
        with open(file, 'w') as f:
            f.write('datetime: ' + datetime.now().strftime('%y%m%d%H%M'))

    runner = Runner(args)
    runner.run()

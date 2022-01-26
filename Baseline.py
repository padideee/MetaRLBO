import argparse
import torch
import Baseline_AMP as amp
import Baseline_RNA14 as rna14
import Baseline_Ising20 as ising20
import Baseline_Ising50 as ising50





parser = argparse.ArgumentParser(description='Argument Parser for Baselines.')
parser.add_argument('--task', type=str, default='AMP',
                    help='problem setting: AMP, Ising20, Ising50, RNA14')
parser.add_argument('--method', type=str, default='dynappo', 
                    help='Method: dynappo, cmaes, genetic, random, adalead')

parser.add_argument('--nModelQueries', type=int, default=4000, help='number of model queries')

args = parser.parse_args()





print(f"Running: {args.task} and method: {args.method}")
if args.task == 'AMP':
    amp.run(args)
elif args.task == 'Ising20':
    ising20.run(args)
elif args.task == 'Ising50':
    ising50.run(args)
elif args.task == 'RNA14':
    rna14.run(args)
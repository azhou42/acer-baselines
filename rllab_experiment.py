# -*- coding: utf-8 -*-
import argparse
import os
import csv
# import platform
import gym
import torch
from torch import multiprocessing as mp

from model import ActorCritic
from optim import SharedRMSprop
from train import train
from test import test
from utils import Counter

from sac.misc.instrument import run_sac_experiment
from rllab.misc.instrument import VariantGenerator

parser = argparse.ArgumentParser(description='ACER')
parser.add_argument('--num-processes', type=int, default=6, metavar='N', help='Number of training async agents (does not include single validation agent)')
parser.add_argument('--T-max', type=int, default=500000, metavar='STEPS', help='Number of training steps')
parser.add_argument('--t-max', type=int, default=100, metavar='STEPS', help='Max number of forward steps for A3C before update')
parser.add_argument('--max-episode-length', type=int, default=500, metavar='LENGTH', help='Maximum episode length')
parser.add_argument('--hidden-size', type=int, default=32, metavar='SIZE', help='Hidden size of LSTM cell')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--on-policy', action='store_true', help='Use pure on-policy training (A3C)')
parser.add_argument('--memory-capacity', type=int, default=100000, metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--replay-ratio', type=int, default=4, metavar='r', help='Ratio of off-policy to on-policy updates')
parser.add_argument('--replay-start', type=int, default=20000, metavar='EPISODES', help='Number of transitions to save before starting off-policy training')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--trace-decay', type=float, default=1, metavar='λ', help='Eligibility trace decay factor')
parser.add_argument('--trace-max', type=float, default=10, metavar='c', help='Importance weight truncation (max) value')
parser.add_argument('--trust-region', action='store_true', help='Use trust region')
parser.add_argument('--trust-region-decay', type=float, default=0.99, metavar='α', help='Average model weight decay rate')
parser.add_argument('--trust-region-threshold', type=float, default=1, metavar='δ', help='Trust region threshold value')
parser.add_argument('--reward-clip', action='store_true', help='Clip rewards to [-1, 1]')
parser.add_argument('--lr', type=float, default=0.0007, metavar='η', help='Learning rate')
parser.add_argument('--lr-decay', action='store_true', help='Linearly decay learning rate to 0')
parser.add_argument('--rmsprop-decay', type=float, default=0.99, metavar='α', help='RMSprop decay factor')
parser.add_argument('--batch-size', type=int, default=16, metavar='SIZE', help='Off-policy batch size')
parser.add_argument('--entropy-weight', type=float, default=0.0001, metavar='β', help='Entropy regularisation weight')
parser.add_argument('--max-gradient-norm', type=float, default=40, metavar='VALUE', help='Gradient L2 normalisation')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=25000, metavar='STEPS', help='Number of training steps between evaluations (roughly)')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--render', action='store_true', help='Render evaluation agent')
parser.add_argument('--name', type=str, default='results', help='Save folder')


def run_acer(variant):
  # BLAS setup
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'

  # Setup
  # args = parser.parse_args()
  # Creating directories.
  save_dir = os.path.join('results', 'results')  
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)  
  print(' ' * 26 + 'Options')

  """
  # Saving parameters
  with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
    for k, v in vars(args).items():
      print(' ' * 26 + k + ': ' + str(v))
      f.write(k + ' : ' + str(v) + '\n')
  """
  # args.env = 'CartPole-v1'  # TODO: Remove hardcoded environment when code is more adaptable
  # mp.set_start_method(platform.python_version()[0] == '3' and 'spawn' or 'fork')  # Force true spawning (not forking) if available
  torch.manual_seed(variant['seed'])
  T = Counter()  # Global shared counter
  # gym.logger.set_level(gym.logger.ERROR)  # Disable Gym warnings

  # Create shared network
  env = gym.make(variant['env'])
  shared_model = ActorCritic(env.observation_space, env.action_space, variant['hidden_size']) 
  shared_model.share_memory()
  """
  if args.model and os.path.isfile(args.model):
    # Load pretrained weights
    shared_model.load_state_dict(torch.load(args.model))
  """
  # Create average network
  shared_average_model = ActorCritic(env.observation_space, env.action_space, variant['hidden_size'])
  shared_average_model.load_state_dict(shared_model.state_dict())
  shared_average_model.share_memory()
  for param in shared_average_model.parameters():
    param.requires_grad = False
  # Create optimiser for shared network parameters with shared statistics
  optimiser = SharedRMSprop(shared_model.parameters(), lr=variant['lr'], alpha=0.99)
  optimiser.share_memory()
  env.close()

  fields = ['t', 'rewards', 'avg_steps', 'time']
  with open(os.path.join(save_dir, 'test_results.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
  # Start validation agent
  processes = []
  p = mp.Process(target=test, args=(0, variant, T, shared_model))
  p.start()
  processes.append(p)

  if not variant['evaluate']:
    # Start training agents
    for rank in range(1, variant['num-processes'] + 1):
      p = mp.Process(target=train, args=(rank, variant, T, shared_model, shared_average_model, optimiser))
      p.start()
      print('Process ' + str(rank) + ' started')
      processes.append(p)

  # Clean up
  for p in processes:
    p.join()

COMMON_PARAMS = {
    'seed': [2 + 10*i for i in range(5)],
    'hidden_size': 32,
    'num-processes': 6,
    'T-max': 5000000,
    't_max': 100,
    'max-episode-length': 1000,
    'on-policy': False,
    'memory_capacity': 100000,
    'replay_ratio': 4,
    'replay_start': 20000,
    'discount': 0.99,
    'trace_decay': 1,
    'trace_max': 10,
    'trust_region': False,
    'trust_region_decay': 0.99,
    'trust_region_threshold': 1,
    'reward_clip': False,
    'lr': 0.0007,
    'lr_decay': False,
    'rmsprop_decay': 0.99,
    'batch_size': 16,
    'entropy_weight': 0.0001,
    'max_gradient_norm': 40,
    'evaluate': False, # ?
    'evaluation-interval': 1000,
    'evaluation-episodes': 1,
    'render': False,
    'name': 'results'
}

from rllab import config
config.DOCKER_IMAGE = "haarnoja/sac"  # needs psutils
config.AWS_IMAGE_ID = "ami-a3a8b3da"  # with docker already pulled

ENV_PARAMS = {
        'env': ['HalfCheetah-v1', 
                'Walker-v1',
                'Ant-v1',
                'Humanoid-v1']
}
def get_variants():
    env_params = ENV_PARAMS
    params = COMMON_PARAMS
    params.update(env_params)

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg

def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    num_experiments = len(variants)
    print('Launching {} experiments.'.format(num_experiments))

    exp_name = 'acer_test1'
    for i, variant in enumerate(variants):
        print("Experiment: {}/{}".format(i, num_experiments))
        experiment_prefix = 'acer_baselines/' + variant['env'] + '/' 
        experiment_name = (
            exp_name + '-' + str(i).zfill(2))

        run_sac_experiment(
            run_acer,
            mode='local',
            variant=variant,
            exp_prefix=experiment_prefix,
            exp_name=experiment_name,
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            snapshot_mode='last',
            snapshot_gap=1000,
            sync_s3_pkl=False,
        )
        return

def main():
    # args = parse_args()
    variant_generator = get_variants()
    launch_experiments(variant_generator)


if __name__ == '__main__':
    main()

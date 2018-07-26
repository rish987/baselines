#!/usr/bin/env python3
import gym;
from baselines.common.cmd_util import mujoco_arg_parser
from baselines.common import set_global_seeds
from runner import resultfile

def train(env_id, seed):
    from baselines.ppo1 import mlp_policy, rand
    set_global_seeds(seed)
    env = gym.make(env_id)
    env.seed(seed)
    result = rand.run(env)
    with open(resultfile, 'a+') as file:
        file.write("{0} result: {1}\n".format(env_id, result));
    env.close()

def main():
    args = mujoco_arg_parser().parse_args()
    train(args.env, seed=args.seed)

if __name__ == '__main__':
    main()

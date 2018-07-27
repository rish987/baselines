#!/usr/bin/env python3
# TODO remove
import sys;
import gym;
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from mpi4py import MPI
from runner import resultfile, graphfile
import pickle 
import numpy as np

def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    # enter a tensorflow session
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, gaussian_fixed_var=True)
    env = make_mujoco_env(env_id, seed)
    pi, result, graph_data = pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    if MPI.COMM_WORLD.Get_rank()==0:
        with open(resultfile, 'a+') as file:
            file.write("{0} result: {1}\n".format(env_id, result));
        with open(graphfile, 'wb') as file:
            pickle.dump([np.array(l) for l in list(zip(*graph_data))], file);
    env.close()

def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()

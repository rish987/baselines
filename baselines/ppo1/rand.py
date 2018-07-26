from baselines.common import Dataset
import tensorflow as tf, numpy as np
from collections import deque

def traj_segment_generator(env, ep_horizon):
    e = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    ep_rets = [] # returns of completed episodes in this segment

    while True:
        ac = env.action_space.sample()

        if e > 0 and e % ep_horizon == 0:
            yield ep_rets
            ep_rets = []

        ob, rew, new, _ = env.step(ac)

        cur_ep_ret += rew
        if new:
            ep_rets.append(cur_ep_ret)
            cur_ep_ret = 0
            ob = env.reset()
            e += 1

def run(env):
    ob_space = env.observation_space
    ac_space = env.action_space

    num_eps = 300

    # prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(env, num_eps)

    ep_rets = seg_gen.__next__()

    assert len(ep_rets) == num_eps

    rewmean = np.mean(ep_rets)

    return rewmean

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

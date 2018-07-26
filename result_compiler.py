from runner import resultfile_ppo, resultfile_rand, environments
import gym
import numpy as np

# - get PPO results for all environments -
envs_to_ppo_results = {}
for env in environments:
    envs_to_ppo_results[env] = []
    
with open(resultfile_ppo) as file:
    lines = file.read().splitlines()
    for line in lines:
        env, result = [e.strip() for e in line.split("result:")]
        envs_to_ppo_results[env] = float(result)
# -

# - get random results for all environments -
envs_to_rand_results = {}
with open(resultfile_rand) as file:
    lines = file.read().splitlines()
    for line in lines:
        env, result = [e.strip() for e in line.split("result:")]
        envs_to_rand_results[env] = float(result)
# -

# - get best results for all environments -
envs_to_best_results = {}
for env in environments:
    envs_to_best_results[env] = gym.envs.registry.env_specs[env].\
            max_episode_steps
# -

# - normalize PPO results -
envs_to_norm_ppo_results = {}
for env in environments:
    print(envs_to_ppo_results[env])
    print(envs_to_rand_results[env])
    print(envs_to_best_results[env])
    envs_to_norm_ppo_results[env] = \
        np.mean(\
        (np.array(envs_to_ppo_results[env]) - envs_to_rand_results[env]) \
        / (envs_to_best_results[env]- envs_to_rand_results[env]))
    print("\t" + str(envs_to_norm_ppo_results[env]))
# - 

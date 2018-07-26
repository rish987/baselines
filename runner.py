import subprocess
import sys
import numpy as np
ppo_command_format = \
        ("mpirun -np 6 python3 -m baselines.ppo1.run_mujoco --num-timesteps"
        " 1000000 --env {0} --seed {1}")
rand_command_format = \
    "python3 -m baselines.ppo1.run_rand --env {0} --seed {1}"

resultfile = "result"
graphfile = "graph_data/data"
graphfile_format = "graph_data/{0}_data_{1}"
resultfile_ppo = "ppo_result"
resultfile_rand = "rand_result"

rename_file_command_format = "mv {0} {1}"

environments = ['InvertedPendulum-v2', 'Reacher-v2',\
    'InvertedDoublePendulum-v2', 'HalfCheetah-v2', 'Hopper-v2',\
    'Swimmer-v2', 'Walker2d-v2']

def main(ppo, graph):
    resultfile_dest = resultfile_ppo if ppo else resultfile_rand
    command_format = ppo_command_format if ppo else rand_command_format
    rename_resultfile_command = \
        rename_file_command_format.format(resultfile, resultfile_dest)
    runs = 3 if (ppo and not graph) else 1

    # clear file
    with open(resultfile, 'w+') as file:
        file.close();

    for environment in environments:
        for run in range(runs):
            process = subprocess.Popen(command_format.format(environment, run)\
                .split());
            process.wait();
            rename_graphfile_command = \
                rename_file_command_format.format(graphfile, \
                graphfile_format.format(environment, run))
            process = subprocess.Popen(rename_graphfile_command.split());

    process = subprocess.Popen(rename_resultfile_command.split());
    process.wait();

if __name__ == '__main__':
    ppo = sys.argv[1] == 'ppo'
    graph = sys.argv[2] == 'graph'
    main(ppo, graph)

import subprocess
from baselines.ppo1.run_mujoco import resultfile
command_format = \
    "mpirun -np 6 python3 -m baselines.ppo1.run_mujoco --env {0} --seed {1}"
def main():
    environments = ['InvertedPendulum-v2', 'Reacher-v2',\
            'InvertedDoublePendulum-v2', 'HalfCheetah-v2', 'Hopper-v2',\
            'Swimmer-v2', 'Walker2d-v2']
    # clear file
    with open(resultfile, 'w+') as file:
        file.close();
    for environment in environments:
        for run in range(3):
            process = subprocess.Popen(command_format.format(environment, run)\
                .split());
            process.wait();

if __name__ == '__main__':
    main()

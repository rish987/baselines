import subprocess
import sys
ppo_command_format = \
        ("mpirun -np 6 python3 -m baselines.ppo1.run_mujoco --num-timesteps"
        " 1000000 --env {0} --seed {1}")
rand_command_format = \
    "python3 -m baselines.ppo1.run_rand --env {0} --seed {1}"

resultfile = "result"
resultfile_ppo = "ppo_result"
resultfile_rand = "rand_result"

rename_resultfile_command_format = "mv " + resultfile + " {0}"

environments = ['InvertedPendulum-v2', 'Reacher-v2',\
    'InvertedDoublePendulum-v2', 'HalfCheetah-v2', 'Hopper-v2',\
    'Swimmer-v2', 'Walker2d-v2']

def main(ppo):
    resultfile_dest = resultfile_ppo if ppo else resultfile_rand
    command_format = ppo_command_format if ppo else rand_command_format
    rename_resultfile_command = \
        rename_resultfile_command_format.format(resultfile_dest)
    runs = 3 if ppo else 1

    # clear file
    with open(resultfile, 'w+') as file:
        file.close();

    for environment in environments:
        for run in range(runs):
            process = subprocess.Popen(command_format.format(environment, run)\
                .split());
            process.wait();

    process = subprocess.Popen(rename_resultfile_command.split());
    process.wait();

if __name__ == '__main__':
    ppo = sys.argv[1] == 'ppo'
    main(ppo)

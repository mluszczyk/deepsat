SBATCH_TEMPLATE = """#!/bin/bash -ex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=72:00:00
#SBATCH --mem=16GB
#SBATCH --output="sbatch.out"
#SBATCH --error="sbatch.err"

cd $SLURM_SUBMIT_DIR

module load tools/python/3.6.0
module load libs/lapack
source ../../../setup.sh

python -m lstm_policy {options}
"""


import os.path
import datetime
import subprocess

PROJ_DIR = os.path.expandvars("$SCRATCH/deepsat/")

SERIES_NAME = datetime.datetime.now().strftime("%y-%m-%d-%H%M%S")

SERIES_DIR = os.path.join(PROJ_DIR, "series", SERIES_NAME)


def run_process(exp_name, opts):
    path = os.path.join(SERIES_DIR, exp_name)
    os.makedirs(path)
    sbatch_path = os.path.join(path, "sat-" + exp_name)

    opts_string = ' '.join(["{}={}".format(name, value) for name, value in opts.items()])

    with open(sbatch_path, "w") as f:
        f.write(SBATCH_TEMPLATE.format(options=opts_string))
    subprocess.run(["sbatch", sbatch_path], check=True, cwd=path)


def main():
    CLAUSE_SIZE = 3
    LEARNING_RATE = 0.001
    VARIABLE_NUM = 8
    CLAUSE_NUM = VARIABLE_NUM * 5
    SAMPLES = 4 * VARIABLE_NUM * 10 ** 6
    EMBEDDING_SIZE = 256
    LSTM_STATE_SIZE = 128
    BATCH_SIZE = 128
    POLICY_HIDDEN_LAYERS = 1
    POLICY_HIDDEN_LAYER_SIZE = 4 * VARIABLE_NUM
    SAT_HIDDEN_LAYERS = 1
    SAT_HIDDEN_LAYER_SIZE = 4 * VARIABLE_NUM

    opts = [
        "VARIABLE_NUM", "CLAUSE_NUM", "EMBEDDING_SIZE",
        "LEARNING_RATE", "SAMPLES", "CLAUSE_SIZE", "BATCH_SIZE",
        "POLICY_HIDDEN_LAYERS", "POLICY_HIDDEN_LAYER_SIZE",
        "SAT_HIDDEN_LAYERS", "SAT_HIDDEN_LAYER_SIZE",
        "LSTM_LAYERS", "LSTM_STATE_SIZE"
    ] 

    for LSTM_LAYERS, LSTM_STATE_SIZE in [(1, 128), (2, 64), (2, 96), (2, 128), (3, 64), (3, 96), (3, 128)]:
        exp_name = "num{}-size{}".format(LSTM_LAYERS, LSTM_STATE_SIZE)
        locals_ = locals()
        opts = {key: locals_[key] for key in opts}
        run_process(exp_name, opts)


if __name__ == '__main__':
    main()

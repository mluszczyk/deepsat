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


def main():
    for VARIABLE_NUM in [4, 5, 6, 7, 8]:
        for CLAUSE_NUM in [VARIABLE_NUM * 4, VARIABLE_NUM * 5]:
            for EMBEDDING_SIZE in [64, 96, 128, 128 + 64]:
                for LEARNING_RATE in [0.001, 0.01]:
                    exp_name = "var{}-clause{}-emb{}-lr{}".format(VARIABLE_NUM, CLAUSE_NUM, EMBEDDING_SIZE, LEARNING_RATE)
                    path = os.path.join(SERIES_DIR, exp_name)
                    os.makedirs(path)
                    SAMPLES = 100000000
                    opts = ["VARIABLE_NUM", "CLAUSE_NUM", "EMBEDDING_SIZE", "LEARNING_RATE", "SAMPLES"]
                    locals_ = locals()
                    opts_string = ' '.join(["{}={}".format(name, locals_[name]) for name in opts])
                    sbatch_path = os.path.join(path, "sat-" + exp_name)
                    with open(sbatch_path, "w") as f:
                        f.write(SBATCH_TEMPLATE.format(options=opts_string))
                    subprocess.run(["sbatch", sbatch_path], check=True, cwd=path)


if __name__ == '__main__':
    main()

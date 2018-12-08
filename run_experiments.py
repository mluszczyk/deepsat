import datetime
import json
import os.path
import subprocess

from options import get_massive_policy

PROJ_DIR = os.path.expandvars("$HOME/deepsat")

SERIES_NAME = datetime.datetime.now().strftime("%y-%m-%d-%H%M%S")

SERIES_DIR = os.path.join(PROJ_DIR, "series", SERIES_NAME)

SBATCH_TEMPLATE = """#!/bin/bash -ex
source ../../setup.sh

set +e
read -r -d '' PARAMS << PARAMS
{options}
PARAMS
set -e

echo "$PARAMS"
export DEEPSAT_PARAMS="$PARAMS"
# neptune run --executable 
python {mode}_policy.py
"""


def run_process(exp_name, mode, opts):
    path = os.path.join(SERIES_DIR, exp_name)
    os.makedirs(path)
    sbatch_path = os.path.join(path, "sat-" + exp_name)

    opts_string = json.dumps(opts)

    with open(sbatch_path, "w") as f:
        f.write(SBATCH_TEMPLATE.format(options=opts_string, mode=mode))
    subprocess.run(["bash", sbatch_path], check=True)


def main():
    mode = "neurosat"
    exp_name = "{}".format(mode)
    opts = get_massive_policy()
    run_process(exp_name, mode, opts)


if __name__ == '__main__':
    main()

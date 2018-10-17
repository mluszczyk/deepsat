import datetime
import json
import os.path
import subprocess

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
neptune run --executable lstm_policy.py
"""


def run_process(exp_name, opts):
    path = os.path.join(SERIES_DIR, exp_name)
    os.makedirs(path)
    sbatch_path = os.path.join(path, "sat-" + exp_name)

    opts_string = json.dumps(opts)

    with open(sbatch_path, "w") as f:
        f.write(SBATCH_TEMPLATE.format(options=opts_string))
    subprocess.run(["bash", sbatch_path], check=True)


def main():
    CLAUSE_SIZE = 3
    LEARNING_RATE = 0.001
    VARIABLE_NUM = 8
    EMBEDDING_SIZE = 256
    LSTM_STATE_SIZE = 128
    BATCH_SIZE = 128
    POLICY_HIDDEN_LAYERS = 1
    SAT_HIDDEN_LAYERS = 1
    VARIABLE_NUM = 10
    SR_GENERATOR = True
    NEPTUNE_ENABLED = True

    opts = [
        "VARIABLE_NUM", "CLAUSE_NUM", "EMBEDDING_SIZE",
        "LEARNING_RATE", "SAMPLES", "CLAUSE_SIZE", "BATCH_SIZE",
        "POLICY_HIDDEN_LAYERS", "POLICY_HIDDEN_LAYER_SIZE",
        "SAT_HIDDEN_LAYERS", "SAT_HIDDEN_LAYER_SIZE",
        "LSTM_LAYERS", "LSTM_STATE_SIZE", "NEPTUNE_ENABLED"
    ] 

    for VARIABLE_NUM in [5]:
        for LSTM_LAYERS in [1]:
            CLAUSE_NUM = VARIABLE_NUM * 5
            SAMPLES = 4 * VARIABLE_NUM * 10 ** 6
            POLICY_HIDDEN_LAYER_SIZE = 4 * VARIABLE_NUM
            SAT_HIDDEN_LAYER_SIZE = 4 * VARIABLE_NUM

            exp_name = "lstm{}-varnum{}".format(LSTM_LAYERS, VARIABLE_NUM)
            locals_ = locals()
            opts = {key: locals_[key] for key in opts}
            run_process(exp_name, opts)


if __name__ == '__main__':
    main()

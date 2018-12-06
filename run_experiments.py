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
neptune run --executable {mode}_policy.py
"""


def run_process(exp_name, mode, opts):
    path = os.path.join(SERIES_DIR, exp_name)
    os.makedirs(path)
    sbatch_path = os.path.join(path, "sat-" + exp_name)

    opts_string = json.dumps(opts)

    with open(sbatch_path, "w") as f:
        f.write(SBATCH_TEMPLATE.format(options=opts_string, mode=mode))
    subprocess.run(["bash", sbatch_path], check=True)


def get_default_settings(mode, clause_aggregation):
    MODE = mode
    CLAUSE_AGGREGATION = clause_aggregation
    SR_GENERATOR = False
    VARIABLE_NUM = 8
    NEPTUNE_ENABLED = True

    LEARNING_RATE = 0.001
    BOARD_WRITE_GRAPH = False

    if MODE == "lstm":
        EMBEDDING_SIZE = 256
        LSTM_STATE_SIZE = 128
        BATCH_SIZE = 128
        POLICY_HIDDEN_LAYERS = 1
        SAT_HIDDEN_LAYERS = 1
        LSTM_LAYERS = 1
        CLAUSE_EMBEDDING_ACTIVATION = "sigmoid"

        if CLAUSE_AGGREGATION == "BOW":
            CLAUSE_HIDDEN_SIZES = [256, 256]
            FORMULA_HIDDEN_SIZES = [256, 256]
            LSTM_STATE_SIZE = 256

    elif MODE == "neurosat":
        MIN_VARIABLE_NUM = VARIABLE_NUM
    else:
        assert False

    if SR_GENERATOR:
        CLAUSE_SIZE = VARIABLE_NUM
    else:
        CLAUSE_SIZE = 3

    opts = [
        "VARIABLE_NUM", "CLAUSE_NUM",
        "LEARNING_RATE",
        "CLAUSE_SIZE",
        "NEPTUNE_ENABLED",
        "SR_GENERATOR",
        "BOARD_WRITE_GRAPH"
    ]

    if MODE == "lstm":
        opts += [
            "EMBEDDING_SIZE",
            "SAMPLES",
            "BATCH_SIZE",
            "POLICY_HIDDEN_LAYERS",
            "POLICY_HIDDEN_LAYER_SIZE",
            "SAT_HIDDEN_LAYERS",
            "SAT_HIDDEN_LAYER_SIZE",
            "LSTM_LAYERS",
            "LSTM_STATE_SIZE",
            "CLAUSE_AGGREGATION",
            "CLAUSE_EMBEDDING_ACTIVATION"
        ]
        if CLAUSE_AGGREGATION == "BOW":
            opts += ["CLAUSE_HIDDEN_SIZES", "FORMULA_HIDDEN_SIZES"]
    elif MODE == "neurosat":
        opts += [
            "MIN_VARIABLE_NUM"
        ]
    else:
        assert False

    CLAUSE_NUM = VARIABLE_NUM * 5

    if MODE == "lstm":
        SAMPLES = 4 * VARIABLE_NUM * 10 ** 6 * 4
        POLICY_HIDDEN_LAYER_SIZE = 4 * VARIABLE_NUM
        SAT_HIDDEN_LAYER_SIZE = 4 * VARIABLE_NUM
    elif MODE == "neurosat":
        pass
    else:
        assert False

    locals_ = locals()
    opts = {key: locals_[key] for key in opts}
    return opts


def main():
    mode = "lstm"
    variable_number = 8
    exp_name = "{}-varnum{}".format(mode, variable_number)
    opts = get_default_settings(mode, "LSTM")
    run_process(exp_name, mode, opts)


if __name__ == '__main__':
    main()

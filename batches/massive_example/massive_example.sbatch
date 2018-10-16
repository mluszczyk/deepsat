#!/bin/bash -ex
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --partition=plgrid
#SBATCH --time=10:00
#SBATCH --cpus-per-task=24
#SBATCH --output="sbatch_cpu.out"
#SBATCH --error="sbatch_cpu.err"

cd $SLURM_SUBMIT_DIR

module load plgrid/tools/python/3.6.5
module load plgrid/tools/gcc/7.3.0
# module load plgrid/libs/lapack
PROJ_DIR=/net/archive/groups/plggluna/henryk/sat_solving/
source $PROJ_DIR/sat_venv_cpu/bin/activate
export PYTHONPATH=$PROJ_DIR/deepsat/
export PYTHONUNBUFFERED=1

set +e
read -r -d '' PARAMS << PARAMS
{
    "VARIABLE_NUM": 30,
    "MIN_VARIABLE_NUM": 30,
    "CLAUSE_SIZE": 3,
    "CLAUSE_NUM": 150,
    "EMBEDDING_SIZE": 128,
    "HIDDEN_LAYERS": [128,128],
    "SAMPLES": 20000000,
    "LEARNING_RATE": 0.0001,
    "NEPTUNE_ENABLED": true,
    "BOARD_WRITE_GRAPH": false,
    "LEVEL_NUMBER": 30,
    "SR_GENERATOR": true,
    "RESTORE": false,
    "BATCH_SIZE": 128
}
PARAMS
set -e

# VARIABLE_NUM=10 CLAUSE_SIZE=3 CLAUSE_NUM=50 EMBEDDING_SIZE=128 HIDDEN_LAYERS=[128,128] SAMPLES=100000000 LEARNING_RATE=0.0002


cd ../../deepsat/

echo "$PARAMS"
export DEEPSAT_PARAMS="$PARAMS"
if [ "$OMPI_COMM_WORLD_RANK" = "0" ]
then
    export NEPTUNE_TOKEN_PATH=$PWD
    neptune run  --executable massive_neurosat_policy.py --tag trial_neptune_1
else
    python massive_neurosat_policy.py
fi
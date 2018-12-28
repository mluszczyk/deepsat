#!/bin/bash -l

#########################
# job-array for SR(30)  #
#########################

#SBATCH --job-name=sr-30-gen

# 16 jobs will run in this array at the same time
#SBATCH --array=1-10

# run for five minutes
#              d-hh:mm:ss
#SBATCH --time=0-02:00:00
#SBATCH -p plgrid
#SBATCH -n 24 # 1 CPU cores
#SBATCH -N 1  # number of machines

# you may not place bash commands before the last SBATCH directive

module load plgrid/tools/python/3.6.5
source /net/archive/groups/plggluna/henryk/sat_solving/deepsat_venv/bin/activate

# each job will see a different ${SLURM_ARRAY_TASK_ID}
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
python dump_dataset.py -j ${SLURM_ARRAY_TASK_ID} -c 30 -o 1000000

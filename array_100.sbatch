#!/bin/bash -l

#########################
# job-array for SR(100)  #
#########################

#SBATCH --job-name=sr-100-gen

# 16 jobs will run in this array at the same time
#SBATCH --array=1-100

# run for five minutes
#              d-hh:mm:ss
#SBATCH --time=0-10:00:00
#SBATCH -p plgrid
#SBATCH -n 24 # number of CPU cores

# you may not place bash commands before the last SBATCH directive

module load plgrid/tools/python/3.6.5
source /net/archive/groups/plggluna/henryk/sat_solving/deepsat_venv/bin/activate

# each job will see a different ${SLURM_ARRAY_TASK_ID}
echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}
python dump_dataset.py -j ${SLURM_ARRAY_TASK_ID} -c 100 -o 150000

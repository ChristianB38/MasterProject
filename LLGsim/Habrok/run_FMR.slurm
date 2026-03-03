#!/bin/bash
#SBATCH --job-name=Hdrive
#SBATCH --output=output_%j.txt
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40

module load Python/3.11.5-GCCcore-13.2.0

source /home2/s4459105/masterproject/myenv/bin/activate

python --version

echo "running LLGsim_FMR.py"
python LLGsim_FMR.py


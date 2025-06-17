#!/bin/bash
#SBATCH --account=meidache_1073         #  ^f^p your actual project/account
#SBATCH --partition=gpu                 #  ^f^p or whichever partition you use
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8               # adjust as needed
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --time=5:00:00                   # adjust walltime as needed
#SBATCH --job-name=splat_all_datasets
#SBATCH --output=logs/splat_%j.out       # stdout  ^f^r logs/splat_<JOBID>.out
#SBATCH --error=logs/splat_%j.err        # stderr  ^f^r logs/splat_<JOBID>.err

INPUT_LOCATION="/project/meidache_1073/butian/data/lerf_ovs/dataset"    # each subfolder is one COLMAP dataset
OUTPUT_LOCATION="/project/meidache_1073/butian/data/models"             # where results go
CONDA_ENV_NAME="splat-distiller"                # your Conda env name


#  ^t^` ^t^` Begin Conda initialization  ^t^` ^t^`
__conda_setup="$('/apps/conda/miniforge3/24.3.0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh" ]; then
        . "/apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh"
    else
        export PATH="/apps/conda/miniforge3/24.3.0/bin:$PATH"
    fi
fi
unset __conda_setup
#  ^t^` ^t^` End Conda initialization  ^t^` ^t^`


conda activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "ERROR: failed to activate Conda environment '$CONDA_ENV_NAME'"
    exit 1
fi


export LD_LIBRARY_PATH="/home1/butianxi/.conda/envs/splat-distiller/lib:$LD_LIBRARY_PATH"


# 3. Ensure log directory exists
mkdir -p logs


# 4. Run the evaluation
python benchmark.py --lerf_ovs /project/meidache_1073/butian/data/lerf_ovs --output_path /project/meidache_1073/butian/data/results/3DGS --training_method 3DGS

echo "All datasets have finished successfully for 3DGS."
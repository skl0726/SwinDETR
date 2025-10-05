#!/bin/sh
#SBATCH -J train_swin_t         # job name
#SBATCH -o train_swin_t.out     # name of stdout output file (%j expands to %jobId)
#SBATCH -e train_swin_t.err     # define file name of standard error
#SBATCH -t 1-00:00:00           # run time (hh:mm:ss)

#SBATCH -p titanxp              # queue or partiton name
#SBATCH -N 1                    # total number of needed computing nodes
#SBATCH -n 6                    # number of nodes (total number of needed processes)
#SBATCH --gres=gpu:2            # gpus per node


cd $SLURM_SUBMIT_DIR

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION=$CUDA_VERSION"


module load cuda/11.2
module load cuDNN/cuda/11.2/8.1.0.77


echo "Start"
echo "condaPATH"

echo "source(path): $HOME/anaconda3/etc/profile.d/conda.sh"
source $HOME/anaconda3/etc/profile.d/conda.sh # path


echo "conda activate gpu"
conda activate gpu # conda env to use

python3 $HOME/temp/SwinDETR/train_swin_t.py


date

echo "conda deactivate"
conda deactivate # deactivate


squeue --job $SLURM_JOBID


echo "##### END #####"
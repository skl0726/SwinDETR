#!/bin/sh
#SBATCH -J swindetr_hdistill25       # job name
#SBATCH -o swindetr_hdistill25.out   # name of stdout output file (%j expands to %jobId)
#SBATCH -e swindetr_hdistill25.err   # define file name of standard error
#SBATCH -t 1-00:00:00                # run time (hh:mm:ss)

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


echo "conda activate swindetr"
conda activate swindetr # conda env to use


cd $HOME/myproject/SwinDETR_D_Q
python3 -m train_ddp --num_distill_query 25 --task_name 'swindetr_hdistill25'


date


echo "conda deactivate"
conda deactivate # deactivate


squeue --job $SLURM_JOBID


echo "##### END #####"
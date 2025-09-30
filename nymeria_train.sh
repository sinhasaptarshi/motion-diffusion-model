#!/bin/bash

#SBATCH --job-name=nymeria_evaluate      # A name for your job
#SBATCH --output=nymeria_eval_%j.txt # Standard output and error log
#SBATCH --error=nymeria_eval_%j.txt   # %j expands to the job ID
#SBATCH --nodes=1                     # Request 1 node   
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2            # Request 1 CPU per task
#SBATCH --time=24:00:00               # Wall-clock time limit (HH:MM:SS)
#SBATCH --mem=400G                      # Memory per node (or per CPU, if --mem-per-cpu is used)
#SBATCH --partition=workq

# source /etc/profile.d/modules.sh
# module load singularitypro/4.1.7 
cd /home/s5a/ve22636.s5a/motion-diffusion-model
source ~/anaconda3/bin/activate
conda activate mdm
python -m train.train_mdm --save_dir save_latest/my_humanml_trans_dec_bert_512_nymeria_final_V4 --dataset humanml --diffusion_steps 50 --arch trans_dec --text_encoder_type bert --mask_frames --use_ema --wandb_name Nymeria_train_latest_v2 --overwrite --num_steps 5000000
# python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512_nymeria_scratch_new_splits_lr5e-6 --dataset humanml --overwrite --wandb_name Nymeria_train_from_scratch_lr5e-6 --eval_during_training

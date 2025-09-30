#!/bin/bash

#SBATCH --job-name=nymeria_evaluate      # A name for your job
#SBATCH --output=nymeria_eval_%j.txt # Standard output and error log
#SBATCH --error=nymeria_eval_%j.txt   # %j expands to the job ID
#SBATCH --nodes=1                     # Request 1 node   
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2            # Request 1 CPU per task
#SBATCH --time=14:00:00               # Wall-clock time limit (HH:MM:SS)
#SBATCH --mem=400G                      # Memory per node (or per CPU, if --mem-per-cpu is used)
#SBATCH --partition=workq

# source /etc/profile.d/modules.sh
# module load singularitypro/4.1.7
cd /lus/lfs1aip2/home/s5a/ve22636.s5a/
singularity exec -w --nv -B /projects/s5a/public:/mnt -B /lus/lfs1aip2/home/s5a/ve22636.s5a/:/home/s5a/ve22636.s5a /lus/lfs1aip2/home/s5a/ve22636.s5a/images/motion bash -c "
cd motion-diffusion-model
source ../anaconda3/bin/activate
conda activate mdm
python -m train.train_mdm --save_dir save_latest/my_humanml_trans_dec_bert_512_hot3d_add --dataset HOT3D --diffusion_steps 50 --arch trans_dec --text_encoder_type bert --mask_frames --use_ema --wandb_name HOT3D_train_add --lr 1e-6 --overwrite --log_interval 100 --save_interval 1000 --combine_conds add
"
# python -m train.train_mdm --save_dir save_latest/my_humanml_trans_dec_bert_512_hdepic_v4_CA_lossawaresampler_temporal_weighted --dataset HDEPIC --diffusion_steps 50 --arch trans_dec --text_encoder_type bert --mask_frames --use_ema --wandb_name HDEPIC_train_latest_lossawaresampler_temporal_weighted --lr 1e-6 --overwrite 
# python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512_nymeria_scratch_new_splits_lr5e-6 --dataset humanml --overwrite --wandb_name Nymeria_train_from_scratch_lr5e-6 --eval_during_training
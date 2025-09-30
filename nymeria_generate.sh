#!/bin/bash

#SBATCH --job-name=nymeria_evaluate      # A name for your job
#SBATCH --output=nymeria_gen_%j.txt # Standard output and error log
#SBATCH --error=nymeria_gen_%j.txt   # %j expands to the job ID
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --ntasks=1   
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2             # Request 1 CPU per task
#SBATCH --time=3:00:00               # Wall-clock time limit (HH:MM:SS)
#SBATCH --mem=200G                      # Memory per node (or per CPU, if --mem-per-cpu is used)
#SBATCH --partition=workq

# source /etc/profile.d/modules.sh
# module load singularitypro/4.1.7 
cd /home/acb11496tf/text-to-motion
source ~/anaconda3/bin/activate
conda activate mdm
# python -m sample.generate --model_path save_latest/my_humanml_trans_dec_bert_512_nymeria_final_V2/model000600000.pt --num_samples 2500 --num_repetitions 3
# python -m sample.generate --model_path save_latest/my_humanml_trans_dec_bert_512_hdepic_v2_wo_text/model000050000.pt --num_samples 2500 --num_repetitions 3 --dataset_name HDEPIC
python -m sample.generate --model_path save_latest/my_humanml_trans_dec_bert_512_nymeria_final_V2/model000800000.pt --num_samples 2500 --num_repetitions 3 --dataset HDEPIC
# singularity exec -w --nv ../images/human_motion bash -c "source ~/anaconda3/bin/activate && conda activate mdm && python train_tex_mot_match.py --name text_mot_match --gpu_id 0 --batch_size 512 --dataset_name t2m --wandb_name HumanML3D_T2M_train_v2"


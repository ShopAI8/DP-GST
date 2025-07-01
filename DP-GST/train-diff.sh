export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
#   --training_split "/your/tsp500_train_concorde.txt" \/your/storage/path
python -u train.py \
  --task "gst" \
  --model "diff" \
  --wandb_logger_name "GST_diffusion_dblp_gamma_3_epsilon_1-delta_0.05_train" \
  --diffusion_type "categorical" \
  --diffusion_schedule "cosine" \
  --diffusion_steps 1000 \
  --do_train \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "./" \
  --training_split "../data/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/train_split.txt" \
  --validation_split "../data/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/valid_split.txt" \
  --test_split "../data/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/test_split.txt" \
  --batch_size 3 \
  --num_epochs 15 \
  --validation_examples 64 \
  --able_wandb \

export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
#   --training_split "/your/tsp500_train_concorde.txt" \/your/storage/path
python -u train.py \
  --task "gst" \
  --model "diff" \
  --wandb_logger_name "GST_diffusion_dblp_gamma_3_epsilon_1-delta_0.05_test" \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "./" \
  --cost_path "../data/result/dblp-g3-epsilon1-delta0.05/dblp-g3-cosine.txt" \
  --test_split "../data/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/test_split.txt" \
  --batch_size 1 \
  --num_epochs 15 \
  --inference_diffusion_steps 5 \
  --sequential_sampling 4 \
  --inference_schedule "cosine" \
  --ckpt_path "../difusco/models/categorical-cosine-dblp-g3-epsilon1-delta0.05-task30/sppdv347/checkpoints/GST-epoch=06-val_solved_cost_mean=0.0000.ckpt" \
  --able_wandb \
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u train.py \
  --task "gst" \
  --model "gcn" \
  --wandb_logger_name "GST_gcn_dblp_gamma_3_epsilon_1-delta_0.05_test" \
  --do_test \
  --n_layers_GCN 8 \
  --hidden_dim_GCN 256 \
  --mlp_layers_GCN 1 \
  --loss_type_GCN "FL" \
  --aggregation_GCN 'mean' \
  --learning_rate_GCN 1e-5 \
  --storage_path "/your/storage/path" \
  --cost_path "/your/result/dblp-g3-gcn-epsilon1-delta0.05/orkut-g3.txt" \
  --test_split "/your/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/test_split.txt" \
  --batch_size 1 \
  --num_epochs 15 \
  --ckpt_path "/your/models/ckpt_path/best.ckpt" \
  --able_wandb \
  

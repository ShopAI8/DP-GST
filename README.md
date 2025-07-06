# DP-GST:Diffusion-based Group Steiner Tree Search with Differential Privacy
DP-GST is a **diffusion model** leveraging **Graph Neural Networks (GNNs)** as its backbone, designed for solving the **Group Steiner Tree (GST)** problem under **differential privacy** constraints.​
#  Code File structures
- `data/`: datasets and associated files, PrunedDP++ algorithm​
- `data/PrunedDP++/`:  PrunedDP++ algorithm​
- `data/PrunedDP++/GST_data.cpp`: a ​python file for generating noise-free dataset and implementing PrunedDP++ 
- `data/PrunedDP++/rucgraph`: dependencies
- `data/process_edge.py`: ​a ​python file for processing bidirectional edges​ in datasets
- `data/gst_data_noise.py`: a ​python file for generating noisy data
- `DP-GST/co_datasets/gst_dataset.py`: a python file for preprocessing the GST dataset
- `DP-GST/models/`: python files for building models
- `DP-GST/models/gnn_encoder.py`:  a python file for building a graph neural network for diffusion models
- `DP-GST/models/gcn_layers.py`: a python file for building network layer features for Att-GCRN
- `DP-GST/models/gcn_model.py`: a python file for building a graph neural network for Att-GCRN
- `DP-GST/utils/diffusion_schedulers.py`: schedulers for denoising diffusion probabilistic models
- `DP-GST/utils/lr_schedulers.py`: a python file for implementing a learning rate scheduler
- `DP-GST/pl_meta_model.py`: a meta PyTorch Lightning model for training and evaluating DP-GST models.
- `DP-GST/pl_gst_model.py`: lightning module for training and evaluating the DP-GST model.
- `DP-GST/pl_gcn_model.py`: lightning module for training and evaluating Att-GCRN models.
- `DP-GST/train.py`: processors for training and evaluating
#  Dependencies
The environment is as follows. Configure your environment using the commands below:
```
conda env create -f environment.yml
conda activate difusco
```
#  Dataset
You can download a real graph dataset from the Stanford Network Analysis Project ([SNA 2025](https://snap.stanford.edu/data/index.html)). Then you can run the program `data/PrunedDP++/GST_data.cpp` to extract noise-free subgraphs from the downloaded dataset and log the GST solutions on these subgraphs.​ 
```
cd data/PrunedDP++
cmake --build ./build
./build/GST_data
```
After obtaining the noise-free dataset, generate the noisy dataset by running the following command. First, remove bidirectional edges from the subgraph.​
```
cd data
python -u process_edge.py \
 --do_file \
 --source_file "/your/source_file/dblp_data_g3_1k.txt" \
 --target_file "/your/target_file/dblp_data_g3_1k_new.txt"\
```
Then, generate the noisy dataset:
```
python -u gst_data_noise.py \
  --gam 3 \
  --epsilon 1 \
  --delta 0.05 \
  --num_train 1600 \
  --num_test_val 40 \
  --num_query_task 30 \
  --dataname "dblp" \
  --source_file "/your/source_file/dblp_data_g3_1k_new.txt"\
  --target_file "/your/target_file/"
  --seed 1234
  ```
#  Reproduce
We provide implementations for three algorithms: **PrunedDP++**,** Att-GCRN**, and **DP-GST**. Reproduce results using these commands. To train your own dataset, modify the path below and adjust model parameters as needed.​
## PrunedDP++

## Att-GCRN
Training on DBLP
 ```
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u train.py \
  --task "gst" \
  --model "gcn" \
  --wandb_logger_name "GST_gcn_dblp_gamma_3_epsilon_1-delta_0.05_train" \
  --do_train \
  --n_layers_GCN 8 \
  --hidden_dim_GCN 256 \
  --mlp_layers_GCN 1 \
  --loss_type_GCN "FL" \
  --aggregation_GCN 'mean' \
  --learning_rate_GCN 1e-5 \
  --storage_path "/your/storage/path" \
  --training_split "/your/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/train_split.txt" \
  --validation_split "/your/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/valid_split.txt" \
  --test_split "/your/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/test_split.txt" \
  --batch_size 3 \
  --num_epochs 15 \
  --validation_examples 64 \
  --able_wandb \
```
Testing on DBLP
 ```
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
```
## DP-GST
Training on DBLP
 ```
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
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
  --storage_path "/your/storage/path" \
  --training_split "/your/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/train_split.txt" \
  --validation_split "/your/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/valid_split.txt" \
  --test_split "/your/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/test_split.txt" \
  --batch_size 3 \
  --num_epochs 15 \
  --validation_examples 64 \
  --able_wandb \
```
Testing on DBLP
```
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u train.py \
  --task "gst" \
  --model "diff" \
  --wandb_logger_name "GST_diffusion_dblp_gamma_3_epsilon_1-delta_0.05_test" \
  --diffusion_type "categorical" \
  --do_test \
  --learning_rate 0.0002 \
  --weight_decay 0.0001 \
  --lr_scheduler "cosine-decay" \
  --storage_path "/your/storage/path" \
  --cost_path "/your/result/dblp-g3-epsilon1-delta0.05/dblp-g3-cosine.txt" \
  --test_split "/your/tasks_30-num_2000-epsilon_1-delta_0.05/dblp_GST1k-gamma_3-noise/test_split.txt" \
  --batch_size 1 \
  --num_epochs 15 \
  --inference_diffusion_steps 5 \
  --sequential_sampling 4 \
  --inference_schedule "cosine" \
  --ckpt_path "/your/models/ckpt_path/best.ckpt" \
  --able_wandb \
```

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
python -u clean.py \
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

"""Lightning module for training and evaluating Att-GCRN models."""
import os
import time
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from torch_geometric.data import DataLoader as GraphDataLoader
from torch_geometric.data import Batch
from pytorch_lightning.utilities import rank_zero_info
import networkx as nx
from models.gcn_model import ResidualGatedGCNModel
from utils.lr_schedulers import get_schedule_fn
from sklearn.utils.class_weight import compute_class_weight 
from co_datasets.gst_dataset import GSTDataset

class GcnModel(pl.LightningModule):
  def __init__(self,
               param_args,
               ):
    super(GcnModel, self).__init__()
    self.args = param_args
    self.sparse = self.args.sparse
    self.loss_type = self.args.loss_type_GCN
    out_channels=1

    self.model = ResidualGatedGCNModel(
        node_dim = self.args.node_dim_GCN,
        hidden_dim=self.args.hidden_dim_GCN,
        num_layers=self.args.n_layers_GCN,
        mlp_layers = self.args.mlp_layers_GCN,
        aggregation=self.args.aggregation_GCN,
        out_channels=out_channels,
        sparse=self.sparse,
        use_activation_checkpoint=self.args.use_activation_checkpoint,
    )
    self.num_training_steps_cached = None
    self.val_solved_costs = []
    self.average_val_solved_costs = []
    self.pred_costs = []
    self.time = []
    if self.args.do_train == True:
            self.train_dataset = GSTDataset(
                data_file=os.path.join(self.args.storage_path, self.args.training_split),
                sparse=self.args.sparse
            )
            
            self.validation_dataset = GSTDataset(
                data_file=os.path.join(self.args.storage_path, self.args.validation_split),
                sparse=self.args.sparse
            )
    elif self.args.do_test == True:
        self.test_dataset = GSTDataset(
            data_file=os.path.join(self.args.storage_path, self.args.test_split),
            sparse=self.args.sparse
        )

  def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges,
                edge_cw, loss_type , gamma = 1,edge_index=None):
    return self.model(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges,edge_cw,loss_type = loss_type,edge_index=edge_index)

  def change_edge(self,label,edge_index,edge_attr,edge_weights,edge_weights_noise):
      dim_x = label.shape[0]
      if edge_index.shape[1]%dim_x == 0:
        return edge_index,edge_attr,edge_weights,edge_weights_noise
      else:
        add_num = dim_x - edge_index.shape[1]%dim_x
        loop_edges = torch.tensor([[0], [0]], 
                                dtype=edge_index.dtype,
                                device=edge_index.device).repeat(1, add_num)
        new_edge_index = torch.cat([edge_index, loop_edges], dim=1)
        def _pad(arr, add_num):
          return torch.cat([
            arr, 
            torch.zeros(add_num, dtype=arr.dtype, device=arr.device)
            ])
        new_edge_attr = _pad(edge_attr, add_num)
        new_weights = _pad(edge_weights, add_num)
        new_weights_noise = _pad(edge_weights_noise, add_num)
        
        return new_edge_index, new_edge_attr, new_weights, new_weights_noise
  
  def training_step(self, batch, batch_idx):
    graph_data = batch
    edge_attr = graph_data.edge_attr
    labels = graph_data.x
    edge_index = graph_data.edge_index
    edge_weights_noise = graph_data.edge_weights_noise
    edge_weights = graph_data.edge_weights
    
    edge_index,edge_attr,edge_weights,edge_weights_noise = self.change_edge(labels,edge_index,edge_attr,edge_weights,edge_weights_noise)

    res_matrix = edge_attr.unsqueeze(-1)
    adj_matrix=res_matrix
    labels = labels.unsqueeze(-1)
    edge_weights_noise=edge_weights_noise.unsqueeze(-1)
    edge_index = edge_index.long().to(res_matrix.device).reshape(2, -1)

    # Calculation of category weights
    edge_labels = adj_matrix.cpu().numpy().flatten()
    edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
    edge_cw = torch.tensor(edge_cw, dtype=torch.float32).to(res_matrix.device)

    y_preds, loss = self.forward(None, edge_weights_noise.float().to(res_matrix.device), labels.float().to(res_matrix.device), None, adj_matrix.float().to(res_matrix.device),
                              edge_cw, self.loss_type, 
                              edge_index=edge_index)
        
    self.log('train/loss', loss, prog_bar=True)
    return loss

  def test_step(self, batch, batch_idx, split='test'):
    start = time.perf_counter()
    graph_data = batch
    device = graph_data.x.device
    edge_attr = graph_data.edge_attr
    labels = graph_data.x
    edge_index = graph_data.edge_index
    edge_weights_noise = graph_data.edge_weights_noise
    edge_weights = graph_data.edge_weights
    res_cost = graph_data.res_cost

    edge_index,edge_attr,edge_weights,edge_weights_noise = self.change_edge(labels,edge_index,edge_attr,edge_weights,edge_weights_noise)

    adj_matrix = edge_attr.unsqueeze(-1)
    labels = labels.unsqueeze(-1)
    edge_index = edge_index.reshape((2, -1))
    edge_weights_noise=edge_weights_noise.unsqueeze(-1)

    min_cost = float('inf')
    res_cost=res_cost.item()

    # Calculation of category weights
    edge_labels = adj_matrix.cpu().numpy().flatten()
    edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
    edge_cw = torch.tensor(edge_cw, dtype=torch.float32).to(self.device)

    for sam in range(1,self.args.sequential_sampling+1):
      y_preds,loss = self.forward(None, edge_weights_noise.float().to(device), labels.float().to(device), None, adj_matrix.float().to(device),
                              edge_cw, self.loss_type, 
                              edge_index=edge_index.long().to(device))
      y_preds = torch.sigmoid(y_preds)
      adj_mat = y_preds.float().cpu().detach().numpy() + 1e-6
      pred_cost=self.find_minimum_spanning_subgraph(adj_mat,edge_index,labels,edge_weights)

      if pred_cost<min_cost:
          min_cost=pred_cost
    
    diff = abs((abs(min_cost)-abs(res_cost))/res_cost)
    print(f"error :{diff}")
    end = time.perf_counter()
    if split == 'test':
      test_time = end - start
      self.time.append(torch.tensor(test_time))
      self.pred_costs.append(torch.tensor([min_cost, res_cost], device=self.device)) 
      metrics = None
    else:
      metrics = {
          f"{split}/ans_loss": diff
      }
    self.val_solved_costs.append(torch.tensor(diff))
    return metrics
  
  def validation_step(self, batch, batch_idx):
    
    return self.test_step(batch, batch_idx, split='val')

  def validation_epoch_end(self, outputs):
    val_solved_cost_mean = torch.mean(torch.stack(self.val_solved_costs))

    if len(self.val_solved_costs) > 0:
        self.log('val_solved_cost_mean', val_solved_cost_mean,prog_bar=True, on_step=False, on_epoch=True,sync_dist=True,reduce_fx='mean')
    metrics = {
        'val_solved_cost_mean': self.trainer.callback_metrics.get('val_solved_cost_mean'),
    }
    print(f'metrics:{metrics}')
    self.val_solved_costs = []

  def test_epoch_end(self, outputs):
        device = self.trainer.strategy.root_device
        dir_path = os.path.dirname(self.args.cost_path)
        os.makedirs(dir_path) if not os.path.exists(dir_path) else print(f"The directory {dir_path} already exists.")

        if self.time:
          time_tensor = torch.stack(self.time).to(device)
          gathered_time_costs = self.trainer.strategy.all_gather(time_tensor)
          if self.trainer.is_global_zero:
              time_mean = torch.mean(gathered_time_costs)
        if self.pred_costs:
          pred_costs_tensor = torch.stack(self.pred_costs).to(device)
          gathered_pred_costs = self.trainer.strategy.all_gather(pred_costs_tensor)
          if self.trainer.is_global_zero:
            all_pred_costs = gathered_pred_costs.view(-1, 2)
            pred_costs_mean = torch.mean(all_pred_costs,dim=0)
            val_solved_cost_mean = abs(pred_costs_mean[0]-pred_costs_mean[1])/pred_costs_mean[1]
        if self.trainer.is_global_zero:
          with open(self.args.cost_path, 'w') as f:
            f.write(f"drop\t{val_solved_cost_mean}\n")
            f.write(f"time_mean\t{time_mean}\n")
            f.write("pred_mean\ttarget_mean:\n")
            f.write(f"{pred_costs_mean[0]}\t{pred_costs_mean[1]}\n")
            f.write("pred\ttarget:\n")
            for row in all_pred_costs:
                f.write(f"{row[0]}\t{row[1]}\n")
  
  def get_total_num_training_steps(self) -> int:
    """Total training steps inferred from datamodule and devices."""
    if self.num_training_steps_cached is not None:
      return self.num_training_steps_cached
    dataset = self.train_dataloader()
    if self.trainer.max_steps and self.trainer.max_steps > 0:
      return self.trainer.max_steps

    dataset_size = (
        self.trainer.limit_train_batches * len(dataset)
        if self.trainer.limit_train_batches != 0
        else len(dataset)
    )

    num_devices = max(1, self.trainer.num_devices)
    effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
    self.num_training_steps_cached = (dataset_size // effective_batch_size) * self.trainer.max_epochs
    return self.num_training_steps_cached

  def configure_optimizers(self):
    rank_zero_info('Parameters: %d' % sum([p.numel() for p in self.model.parameters()]))
    rank_zero_info('Training steps: %d' % self.get_total_num_training_steps())

    if self.args.lr_scheduler == "constant":
      return torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate_GCN, weight_decay=self.args.weight_decay)

    else:
      optimizer = torch.optim.AdamW(
          self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
      scheduler = get_schedule_fn(self.args.lr_scheduler, self.get_total_num_training_steps())(optimizer)

      return {
          "optimizer": optimizer,
          "lr_scheduler": {
              "scheduler": scheduler,
              "interval": "step",
          },
      }

  def collate_fn(batch):
    return Batch.from_data_list(batch)

  def train_dataloader(self):
    batch_size = self.args.batch_size
    train_dataset = torch.utils.data.Subset(self.train_dataset, range(48000))
    train_dataloader = GraphDataLoader(
        train_dataset, batch_size=batch_size, collate_fn=self.collate_fn,shuffle=True,
        num_workers=self.args.num_workers, pin_memory=True,
        persistent_workers=True, drop_last=True)
    return train_dataloader

  def test_dataloader(self):
    batch_size = 1
    print("Test dataset size:", len(self.test_dataset))
    test_dataloader = GraphDataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader

  def val_dataloader(self):
    batch_size = 1
    val_dataset = torch.utils.data.Subset(self.validation_dataset, range(self.args.validation_examples))
    print("Validation dataset size:", len(val_dataset))
    val_dataloader = GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_dataloader

  def find_minimum_spanning_subgraph(self,edge_probs, edge_index, groups,edge_weights):
    """
    Args:
        edge_probs: an array like (E,) representing the probabilities of the edges
        edge_index: an array like (2, E) representing the starting and ending points of the edges
        groups: the group that each node belongs to, as in (V,)

    Returns:
        returns the edges in the minimum spanning tree over all groups
    """
    edge_probs, edge_index, groups, edge_weights = self.to_cpu(edge_probs, edge_index, groups, edge_weights)
    edge_probs = np.squeeze(edge_probs)
    edge_index = np.squeeze(edge_index)
    groups = np.squeeze(groups)
    edge_weights = np.squeeze(edge_weights)
    edge_index = edge_index.long()
    sorted_indices = np.argsort(edge_probs)[::-1]
    sorted_indices = sorted_indices.copy()
    sorted_edges = edge_index[:, sorted_indices]
    sorted_weights = edge_weights[sorted_indices]
    G = nx.Graph()
    unique_groups = np.unique(groups[groups != 0])
    unique_groups=np.sort(unique_groups)
    for i in range(sorted_edges.shape[1]):
      
      u, v = sorted_edges[:, i]
      w=sorted_weights[i]
      u=u.item()
      v=v.item()
      w=w.item()
      G.add_edge(u, v, weight=w)
      connected_components = list(nx.connected_components(G))
      
      for component in connected_components:
        unique_groups_in_component = np.unique(groups[np.array(list(component))])
        covered_groups = np.sort(unique_groups_in_component[unique_groups_in_component != 0])
        if covered_groups.shape == unique_groups.shape:
            
          subgraph = G.subgraph(component)  
          mst = nx.minimum_spanning_tree(subgraph) 
          if self._covers_all_groups(mst, groups,unique_groups):
              
            return self._prune_tree(mst, groups,unique_groups)
            
    return 0  

  def to_cpu(self,*args):
    """
    Moves all tensors in the given arguments to the CPU.
    
    Args:
        *args: Variable length argument list, which can include
            tensors or other data types.
    
    Returns:
        A tuple where each tensor is moved to the CPU if applicable.
    """
    return tuple(arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args)
  def _covers_all_groups(self,subgraph, groups,group_set):
    """
    check if the subgraph covers all groups
    """
    nodes_in_subgraph = subgraph.nodes
    unique_covered_groups = np.unique(groups[np.array(list(nodes_in_subgraph))])
    covered_groups = np.sort(unique_covered_groups[unique_covered_groups != 0])

    return covered_groups.shape == group_set.shape

  def _prune_tree(self,tree, groups,group_set):
    """
    The minimum spanning tree is cropped to remove redundant edges, ensuring that all groups are still covered.
    Also ensure that the cropped tree is still a connected tree.
    """

    edges_sorted = sorted(tree.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    prune_flag = 1
    while prune_flag:
        prune_flag = 0
        for u, v, data in edges_sorted:

            tree.remove_edge(u, v)

            if nx.is_connected(tree):
                if not self._covers_all_groups(tree, groups, group_set):
                    tree.add_edge(u, v, weight=data['weight'])
                else:
                    prune_flag = 1
            else:
                tree.add_edge(u, v, weight=data['weight'])
    total_weight = sum(data['weight'] for u, v, data in tree.edges(data=True))

    return total_weight  

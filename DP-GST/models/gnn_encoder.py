import functools
import math

import torch
import torch.nn.functional as F
from torch import nn
from models.nn import (
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max
import torch.utils.checkpoint as activation_checkpoint

 
class GNNLayer(nn.Module):
  """Configurable GNN Layer
  Implements the Gated Graph ConvNet layer:
      h_i =h_i + ReLU ( Norm(U*h_i + Aggr.( sigma_ij, V*h_j)) ),
      sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij + D*ew_ij),
      e_ij =e_ij + ReLU ( Norm(A*h_i + B*h_j + C*e_ij + D*ew_ij) ),
      ew_ij = ew_ij + ReLU ( Norm(A*h_i + B*h_j + D*ew_ij) ),
      where Aggr. is an aggregation function: sum/mean/max.
  References:
      - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
      - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
  """

  def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
    """
    Args:
        hidden_dim: Hidden dimension size (int)
        aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
        norm: Feature normalization scheme ("layer"/"batch"/None)
        learn_norm: Whether the normalizer has learnable affine parameters (True/False)
        track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        gated: Whether to use edge gating (True/False)
    """
    super(GNNLayer, self).__init__()
    self.hidden_dim = hidden_dim
    self.aggregation = aggregation
    self.norm = norm
    self.learn_norm = learn_norm
    self.track_norm = track_norm
    self.gated = gated
    assert self.gated, "Use gating with GCN, pass the `--gated` flag"
 
    self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.D = nn.Linear(hidden_dim, hidden_dim, bias=True)  # set for EW
    self.U_ew = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.V_ew = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.norm_h = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

    self.norm_e = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)
    self.norm_ew = {
      "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
      "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

  def forward(self, h, e, graph, ew, mode="residual", edge_index=None, sparse=False):
    """
    Args:
        In Dense version:
          h: Input node features (B x V x H)
          e: Input edge features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          mode: str
        In Sparse version:
          h: Input node features (V x H)
          e: Input edge features (E x H)
          graph: torch_sparse.SparseTensor
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Updated node and edge features
    """
    if not sparse:
      batch_size, num_nodes, hidden_dim = h.shape
    else:
      batch_size = None
      num_nodes, hidden_dim = h.shape
    h_in = h
    e_in = e
    ew_in = ew
    # Linear transformations for node update
    Uh = self.U(h)  # B x V x H
    Uew = self.U_ew(ew)

    if not sparse:
      Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H
    else:
      Vh = self.V(h[edge_index[1]])  # E x H

    # Linear transformations for edge update and gating
    Ah = self.A(h)  # B x V x H, source
    Bh = self.B(h)  # B x V x H, target
    Ce = self.C(e)  # B x V x V x H / E x H
    Dew = self.D(ew)
    Vew = self.V_ew(ew)
    # Update edge features and compute edge gates
    if not sparse:
      ew = (Ah.unsqueeze(1) + Bh.unsqueeze(2) + Dew) * graph.unsqueeze(-1)
      e = ew + Ce * graph.unsqueeze(-1)  # B x V x V x H
    else:
      ew = Ah[edge_index[1]] + Bh[edge_index[0]] + Dew
      e = ew + Ce  # E x H

    gates = torch.sigmoid(e)  # B x V x V x H / E x H
    Vh = Vh + Vew
    # Update node features
    h = Uh + self.aggregate(Vh, graph, gates, edge_index=edge_index, sparse=sparse)  # B x V x H
    ew += Uew
    # Normalize node features
    if not sparse:
      h = self.norm_h(
          h.view(batch_size * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h
    else:
      h = self.norm_h(h) if self.norm_h else h

    # Normalize edge features
    if not sparse:
      e = self.norm_e(
          e.view(batch_size * num_nodes * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e
    else:
      e = self.norm_e(e) if self.norm_e else e

    if not sparse:
      ew = self.norm_ew(
          ew.view(batch_size * num_nodes * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_ew else ew
    else:
      ew = self.norm_ew(ew) if self.norm_ew else ew
    # Apply non-linearity
    h = F.relu(h)
    e = F.relu(e)
    ew = F.relu(ew)
    # Make residual connection
    if mode == "residual":
      h = h_in + h
      e = e_in + e
      ew = ew_in + ew
    return h, e, ew

  def aggregate(self, Vh, graph, gates, mode=None, edge_index=None, sparse=False):
    """
    Args:
        In Dense version:
          Vh: Neighborhood features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          gates: Edge gates (B x V x V x H)
          mode: str
        In Sparse version:
          Vh: Neighborhood features (E x H)
          graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
          gates: Edge gates (E x H)
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Aggregated neighborhood features (B x V x H)
    """
    # Perform feature-wise gating mechanism
    Vh = gates * Vh  # B x V x V x H

    # Aggregate neighborhood features
    if not sparse:
      Vh = Vh * graph.unsqueeze(-1)  # B x V x V x H, only keep edges that exist in the graph
      if (mode or self.aggregation) == "mean":
        # Calculate mean by dividing by the number of neighbors
        neighbor_count = torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh)  # B x V x 1
        neighbor_count = neighbor_count + (neighbor_count == 0).float()  # Prevent division by zero
        return torch.sum(Vh, dim=2) / neighbor_count
      elif (mode or self.aggregation) == "max":
        # Apply masking to Vh where there's no edge
        Vh = Vh.masked_fill(graph.unsqueeze(-1) == 0, float('-inf'))  # Mask out non-edges
        return torch.max(Vh, dim=2)[0]
      else:
        return torch.sum(Vh, dim=2)
    else:
      sparseVh = SparseTensor(
          row=edge_index[0],
          col=edge_index[1],
          value=Vh,
          sparse_sizes=(graph.size(0), graph.size(1))
      )

      if (mode or self.aggregation) == "mean":
        return sparse_mean(sparseVh, dim=1)

      elif (mode or self.aggregation) == "max":
        return sparse_max(sparseVh, dim=1)

      else:
        return sparse_sum(sparseVh, dim=1)

class ScalarEmbeddingLinear(nn.Module):
    def __init__(self, num_pos_feats=64):
        super().__init__()
        # 定义一个线性层，将每条边的信息嵌入到 num_pos_feats 维度
        self.edge_proj = nn.Linear(1, num_pos_feats)

    def forward(self, x):
        # 输入的 x 形状为 (B, V, V)，代表边的信息或邻接矩阵
        B, V, _ = x.shape
        
        # 对 x 扩展维度，变为 (B, V, V, 1)，以便传入线性层处理
        x_embed = x.unsqueeze(-1)  # (B, V, V, 1)

        # 使用线性层对每一条边的信息进行嵌入
        pos_x = self.edge_proj(x_embed)  # (B, V, V, num_pos_feats)

        return pos_x


class EWEmbeddingLinear(nn.Module):
  def __init__(self, num_pos_feats=64):
    super().__init__()
    # 定义一个线性层，将每条边的信息嵌入到 num_pos_feats 维度
    self.edge_proj = nn.Linear(1, num_pos_feats)

  def forward(self, x):
    # 输入的 x 形状为 (B, V, V)，代表边的信息或邻接矩阵
    # B, V, _ = x.shape

    # 对 x 扩展维度，变为 (B, V, V, 1)，以便传入线性层处理
    x_embed = x.unsqueeze(-1)  # (B, V, V, 1)

    # 使用线性层对每一条边的信息进行嵌入
    pos_x = self.edge_proj(x_embed)  # (B, V, V, num_pos_feats)

    return pos_x

class ScalarEmbeddingLinear1D(nn.Module):
    def __init__(self, num_pos_feats=64):
        super().__init__()
        # 定义一个线性层，将标量嵌入到 num_pos_feats 维度
        self.scalar_proj = nn.Linear(1, num_pos_feats)

    def forward(self, x):
        # 假设输入 x 的形状为 (B, V)，表示每个元素的标量值
        x_embed = x.unsqueeze(-1)  # 扩展维度，形状变为 (B, V, 1)

        # 通过线性层将标量嵌入到高维空间
        pos_x = self.scalar_proj(x_embed)  # 形状 (B, V, num_pos_feats)

        return pos_x

def run_sparse_layer(layer, time_layer, out_layer, adj_matrix, edge_index, add_time_on_edge=True):
  def custom_forward(*inputs):
    x_in = inputs[0]
    e_in = inputs[1]
    time_emb = inputs[2]
    x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
    if add_time_on_edge:
      e = e + time_layer(time_emb)
    else:
      x = x + time_layer(time_emb)
    x = x_in + x
    e = e_in + out_layer(e)
    return x, e
  return custom_forward


class GNNEncoder(nn.Module):
  """Configurable GNN Encoder
  """

  def __init__(self, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               sparse=False, use_activation_checkpoint=False, node_feature_only=False,
               *args, **kwargs):
    super(GNNEncoder, self).__init__()
    self.sparse = sparse
    self.node_feature_only = node_feature_only
    self.hidden_dim = hidden_dim
    time_embed_dim = hidden_dim // 2
    self.node_embed = nn.Linear(hidden_dim, hidden_dim)
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
    self.EW_embed = nn.Linear(hidden_dim, hidden_dim)
    
    self.pos_embed = ScalarEmbeddingLinear1D(hidden_dim)
    self.edge_pos_embed = ScalarEmbeddingLinear(hidden_dim)
    self.EW_pos_embed = EWEmbeddingLinear(hidden_dim)
    
    self.time_embed = nn.Sequential(
        linear(hidden_dim, time_embed_dim),
        nn.ReLU(),
        linear(time_embed_dim, time_embed_dim),
    )
    self.out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)

    )

    self.layers = nn.ModuleList([
        GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
        for _ in range(n_layers)
    ])

    self.time_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                time_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ])

    self.per_layer_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])
    self.use_activation_checkpoint = use_activation_checkpoint

  def dense_forward(self,  label, edge_weights , adj_matrix , xt, t):
    """
    Args:
        x: Input node coordinates (B x V x 2)
        graph: Graph adjacency matrices (B x V x V)
        timesteps: Input node timesteps (B)
        edge_index: Edge indices (2 x E)
    Returns:
        Updated edge features (B x V x V)
    """
    x = self.node_embed(self.pos_embed(label))
    e = self.edge_embed(self.edge_pos_embed(xt))#B V V H
    ew = self.EW_embed(self.EW_pos_embed(edge_weights))
    time_emb = self.time_embed(timestep_embedding(t, self.hidden_dim))
    # graph = torch.ones_like(graph).long()

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in, ew_in = x, e , ew

      x, e , ew = layer(x, e, edge_weights,mode="direct",ew=ew)
      e = e + time_layer(time_emb)[:, None, None, :]
      x = x_in + x
      e = e_in + out_layer(e)
      ew = ew_in + ew
    e = self.out(e.permute((0, 3, 1, 2)))
    return e

  def sparse_forward(self,label,edge_weights,edge_index,xt,t):
    """
    Args:
        x: Input node coordinates (V x 2)
        graph: Graph edge features (E)
        timesteps: Input edge timestep features (E)
        edge_index: Adjacency matrix for the graph (2 x E)
    Returns:
        Updated edge features (E x H)
    """
    # Embed edge features
    x = self.node_embed(self.pos_embed(label.unsqueeze(0)).squeeze(0))
    e = self.edge_embed(self.edge_pos_embed(xt.expand(1, 1, -1)).squeeze())
    ew = self.EW_embed(self.EW_pos_embed(edge_weights))
    time_emb = self.time_embed(timestep_embedding(t, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb,ew)
    e = e.reshape((1, x.shape[0], -1, e.shape[-1])).permute((0, 3, 1, 2))
    e = self.out(e).reshape(-1, edge_index.shape[1]).permute((1, 0))
    return e

  def sparse_forward_node_feature_only(self, x, timesteps, edge_index):
    x = self.node_embed(self.pos_embed(x))
    x_shape = x.shape
    e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)
    x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
    x = self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
    return x

  def sparse_encoding(self, x, e, edge_index, time_emb,ew):
    adj_matrix = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=ew,
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    adj_matrix = adj_matrix.to(x.device)

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in ,ew_in = x, e,ew
      x, e ,ew= layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True,ew=ew)
      if not self.node_feature_only:
        e = e + time_layer(time_emb)
      else:
        x = x + time_layer(time_emb)
      x = x_in + x
      ew = ew_in + ew
      e = e_in + out_layer(e)
    return x, e

  def forward(self, label, edge_weights , adj_matrix , xt, t,edge_index=None):
      if self.sparse :
        return self.sparse_forward(label,edge_weights,edge_index,xt,t)
      else :
        return self.dense_forward( label, edge_weights , adj_matrix , xt, t)
import torch
import torch.nn.functional as F
import torch.nn as nn
from models.nn import normalization
import numpy as np

class BatchNormNode(nn.Module):
    """Batch normalization for node features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim,elementwise_affine=True)
    def forward(self, x):
        """
        Args:
            x: Node features (batch_size, hidden_dim, num_nodes)

        Returns:
            x_bn: Node features after batch normalization (batch_size, hidden_dim, num_nodes)
        """
        
        x_bn = self.layer_norm(x)
        return x_bn

class BatchNormEdge(nn.Module):
    """Batch normalization for edge features.
    """

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        
        self.layer_norm = nn.LayerNorm(hidden_dim,elementwise_affine=True)

    def forward(self, e,sparse=False):
        """
        Args:
            e: Edge features (batch_size, hidden_dim, num_nodes, num_nodes)

        Returns:
            e_bn: Edge features after batch normalization (batch_size, hidden_dim, num_nodes, num_nodes)
        """
        
        e_bn = self.layer_norm(e)
        return e_bn


class NodeFeatures(nn.Module):
    """Convnet features for nodes.
    
    Using `sum` aggregation:
        x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]
    
    Using `mean` aggregation:
        x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
    """
    
    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.Us = nn.Linear(hidden_dim, hidden_dim)  # 节点自身变换
        self.Vs = nn.Linear(hidden_dim, hidden_dim)  # 邻居特征变换
    def forward(self, x, edge_gate,edge_index=None):
        """
        Args:
            x: Node features (batch_size, hidden_dim, num_nodes)
            e: Edge features (batch_size, hidden_dim, num_nodes, num_nodes)
            In Sparse version:
            x: Input node features (V x H)
            e: Input edge features (E x H)
        Returns:
            x_new: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
        
        Ux = self.Us(x)  
        Vx = self.Vs(x)  
        
        src_nodes, dst_nodes = edge_index
        dt = edge_gate.dtype
       
        neighbor_Vx = Vx[src_nodes]  # (E, H)
        gated_neighbors = neighbor_Vx * edge_gate  # (E, H)
        
        aggregated = torch.zeros_like(x).to(dt)  # (V, H)
        
        aggregated.index_add_(0, dst_nodes, gated_neighbors)
        
        if self.aggregation == "mean":
            gate_sum = torch.zeros(x.size(0), x.size(1), device=x.device).to(dt)
            gate_sum.index_add_(0, dst_nodes, edge_gate)

            aggregated = aggregated / (gate_sum + 1e-20)
            
        x_new = Ux + aggregated

        return x_new


class EdgeFeatures(nn.Module):
    """Convnet features for edges.

    e_ij = U*e_ij + V*(x_i + x_j)
    """

    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        
        self.Us = nn.Linear(hidden_dim, hidden_dim)  
        self.Vs = nn.Linear(hidden_dim, hidden_dim)  
        
    def forward(self, x, e, edge_index=None):
        """
        Args:
            x: Node features (batch_size, hidden_dim, num_nodes)
            e: Edge features (batch_size, hidden_dim, num_nodes, num_nodes)
            In Sparse version:
            x: Input node features (V x H)
            e: Input edge features (E x H)
        Returns:
            e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        """
       
        Ue = self.Us(e)  
        Vx = self.Vs(x)  
        src_nodes, dst_nodes = edge_index
        
        Vx_agg = Vx[src_nodes] + Vx[dst_nodes]  # (E, H)
        
        e_new = Ue + Vx_agg
        return e_new
 

class ResidualGatedGCNLayer(nn.Module):
    """Convnet layer with gating and residual connection.
    """

    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e,edge_index=None):
        """
        Args:
            x: Node features (batch_size, hidden_dim, num_nodes)
            e: Edge features (batch_size, hidden_dim, num_nodes, num_nodes)
            In Sparse version:
            x: Input node features (V x H)
            e: Input edge features (E x H)

        Returns:
            x_new: Convolved node features (batch_size, hidden_dim, num_nodes)
            e_new: Convolved edge features (batch_size, hidden_dim, num_nodes, num_nodes)
        """
        e_in = e # B x H x V x V
        x_in = x # B x H x V
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in,edge_index)  # B x H x V x V
        # Compute edge gates
        edge_gate = F.sigmoid(e_tmp)
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate,edge_index) # B x H x V
        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        # Residual connection
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new


class MLP(nn.Module):
    """Multi-layer Perceptron for output prediction.
    """

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L

        #Sparse version
        layers = []
        for _ in range(self.L-1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
        self.layers = nn.Sequential(*layers)
        
        self.output = nn.Linear(hidden_dim, output_dim)
        self.out = nn.Sequential(
            normalization(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1, bias=True)
        )
    def forward(self, x_node,x,edge_index=None):
        """
        Args:
            x: Input features (batch_size, hidden_dim, num_nodes, num_nodes)

        Returns:
            y: Output predictions (batch_size, output_dim, num_nodes, num_nodes)
        """
        original_shape = x.shape
        
        if x.dim() == 3:
            x = x.view(-1, x.size(-1))  # (B*N, H)
        
        x = self.layers(x)
        x = x.reshape((1, x_node.shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
        y = self.out(x).reshape(-1, edge_index.shape[1]).permute((1, 0))
        return y

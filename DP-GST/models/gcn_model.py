import torch
import torch.nn.functional as F
import torch.nn as nn
from models.gcn_layers import ResidualGatedGCNLayer, MLP


class ResidualGatedGCNModel(nn.Module):
    """Residual Gated GCN Model for outputting predictions as edge adjacency matrices.

    References:
        Paper: https://arxiv.org/pdf/1711.07553v2.pdf
        Code: https://github.com/xbresson/spatial_graph_convnets
    """

    def __init__(self, node_dim, hidden_dim,num_layers, mlp_layers,aggregation,out_channels=2,sparse=False,use_activation_checkpoint=False, dtypeFloat= torch.cuda.FloatTensor,dtypeLong=torch.cuda.LongTensor):
        super(ResidualGatedGCNModel, self).__init__()
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        self.node_dim = node_dim
        self.voc_edges_in = 3
        self.voc_edges_out = out_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mlp_layers = mlp_layers
        self.aggregation = aggregation
        self.sparse = sparse
        # Node and edge embedding layers/lookups
        self.nodes_coord_embedding = nn.Linear(self.node_dim, self.hidden_dim, bias=False)
        self.edge_embed = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        # Define GCN Layers
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, self.aggregation))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Define MLP classifiers
        self.mlp_edges = MLP(self.hidden_dim, self.voc_edges_out, self.mlp_layers)
        self.use_activation_checkpoint = use_activation_checkpoint

    def forward(self, x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges,
                edge_cw, loss_type = "CE", gamma = 2,edge_index=None):
        batch_size = x_nodes.size(0)
    
        h_nodes = self.nodes_coord_embedding(x_nodes)  # (B, V, H)

        h_edges = self.edge_embed(x_edges_values)  # (B, E, H)

        h_nodes = h_nodes.view(-1, self.hidden_dim)  # (B*V, H)
        h_edges = h_edges.view(-1, self.hidden_dim)  # (B*E, H)
        edge_index = edge_index.view(2, -1)  # (2, B*E)
        # GCN layers
        for layer in range(self.num_layers):
            h_nodes, h_edges = self.gcn_layers[layer](h_nodes, h_edges,edge_index) 
        y_pred_edges = self.mlp_edges(h_nodes,h_edges,edge_index)# (B*E, 1)
        # y_pred_edges = y_pred_edges.view(batch_size, -1, self.voc_edges_out)
        loss = sparse_edge_loss(y_pred_edges,y_edges,edge_cw,loss_type = loss_type, gamma = gamma)
        return y_pred_edges, loss

def sparse_edge_loss(y_pred, y_true, edge_cw=None, 
                    loss_type='CE', reduction='mean', 
                    gamma=2, edge_mask=None):
    """ 
    Sparse graph edge prediction loss function (binary categorization version)
        
    Args: 
        y_pred: predicted value (B, E) or (B, E, 1), raw logits 
        y_true: true label (B, E), range [0, 1] 
        edge_cw: category weight (C,) or scalar, None means equal weight 
        loss_type: loss type ['BCE', 'Focal', 'MSE']
                reduction: aggregation mode ['mean', 'sum', 'none'] 
        gamma: hyperparameter of Focal Loss 
        edge_mask: valid edge mask (B, E), True means valid edges        
    Returns: 
        loss: calculated loss value 
    """
    y_pred = y_pred.view(-1)          # (B*E,) 
    y_true = y_true.view(-1).float()  # (B*E,)
    
    if edge_mask is not None:
        mask = edge_mask.view(-1).bool()  # (B*E,)
        y_pred = y_pred[mask]             # (M,)
        y_true = y_true[mask]             # (M,)
    
    if loss_type == 'CE':
        loss = F.binary_cross_entropy_with_logits(
            y_pred, y_true, 
            weight=edge_cw, 
            reduction=reduction
        )
    elif loss_type == 'FL':
        loss = binary_focal_loss_with_logits(
            y_pred, y_true,
            alpha=edge_cw,
            gamma=gamma,
            reduction=reduction
        )
    elif loss_type == 'MSE':
        loss = F.mse_loss(
            torch.sigmoid(y_pred), y_true,
            reduction=reduction
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return loss

def binary_focal_loss_with_logits(input, target, alpha=0.5, gamma=2.0, reduction='mean'):
    """ 
    Binary Focal Loss Implementation
        
    Args: 
        input: predicted logits (N,) 
        target: true labels (N,) ∈ {0, 1} 
        alpha: category weight balance factor (scalar or tuple) 
        gamma: difficult sample focusing parameter 
        reduction: aggregation mode 
    """
    p = torch.sigmoid(input)
    
    ce_loss = F.binary_cross_entropy_with_logits(
        input, target, reduction='none'
    )
    
    p_t = p * target + (1 - p) * (1 - target)
    
    alpha_neg, alpha_pos = alpha
    # Select alpha_neg or alpha_pos according to the 0/1 value of the target.
    alpha_t = alpha_neg * (1 - target) + alpha_pos * target
    
    
    # Focal Loss
    loss = alpha_t * (1 - p_t)**gamma * ce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

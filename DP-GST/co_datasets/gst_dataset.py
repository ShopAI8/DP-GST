"""GST (Group Stenier Tree) dataset."""
 
import glob
import os

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData

class GSTDataset(torch.utils.data.Dataset):
  def __init__(self, data_file, sparse=1):
    self.data_file = data_file
    self.sparse = sparse
    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    try:
        # Get the line and strip leading/trailing whitespace
        line = self.file_lines[idx].strip()

        #label information that stores which group the vertex belongs to
        label_line = line.split('Label ')[1] 
        label_line2 = label_line.split('|GroupNum')[0]
        label = np.array([int(t) for t in label_line2.split(' ') if t != ''])  

        #weight information, which stores the weights between edges
        graph = label_line.split('Graph ')[1]
        graph2 = graph.split('Result ')[0]
        weights = np.array([int(t) for t in graph2.split(' ') if t != '']) 

        # results for Storage Group Steiner Tree
        tree = graph.split('Result ')[1]
        res = np.array([int(t) for t in tree.split(' ') if t != ''])  
        return label, weights, res

    except (IndexError, ValueError) as e:
        print(f"Error parsing line at index {idx}: {e}")
        return None


  def __getitem__(self, idx):
    label,weights,res = self.get_example(idx)
    if not self.sparse:

      edge_weights = np.zeros((label.shape[0], label.shape[0]))
      edge_weights_noise = np.zeros((label.shape[0], label.shape[0]))
      adj_matrix = np.zeros((label.shape[0], label.shape[0]))
      for i in range(0,weights.shape[0],4):
        edge_weights[weights[i],weights[i+1]] = weights[i+2]
        edge_weights[weights[i+1],weights[i]] = weights[i+2]
        edge_weights_noise[weights[i],weights[i+1]] = weights[i+3]
        edge_weights_noise[weights[i+1],weights[i]] = weights[i+3]
        adj_matrix[weights[i+1],weights[i]] = 1
        adj_matrix[weights[i],weights[i+1]] = 1

      res_matrix = np.zeros((label.shape[0], label.shape[0]))
      for i in range(0,res.shape[0],2):
        res_matrix[res[i],res[i+1]] = 1
        res_matrix[res[i+1],res[i]] = 1

      res_cost_sum = 0
      for i in range(0, res.shape[0], 2):
        res_cost_sum += edge_weights[res[i], res[i+1]]
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          torch.from_numpy(label).float(),
          torch.from_numpy(edge_weights).float(),
          torch.from_numpy(edge_weights_noise).float(),
          torch.from_numpy(adj_matrix).float(),
          torch.from_numpy(res_matrix).float(),
          res_cost_sum/2
      )
    else:
      #The original graph removes half of the bidirectional edges, which have to be added here, so the original weight vector size is divided by 2
      edge_index = np.zeros((2, weights.shape[0] // 2))
      edge_weights_noise = np.zeros(weights.shape[0] // 2)
      edge_weights = np.zeros(weights.shape[0] // 2)
      edge_attr = np.zeros(weights.shape[0] // 2)
      res_box = []
      #Add a bi-directional edge to record the starting index start
      start = int((weights.shape[0]-1)/4)+1
      for idx, i in enumerate(range(0, weights.shape[0], 4)):
          edge_weights_noise[idx] = weights[i+3]
          edge_weights[idx] = weights[i+2]

          edge_index[0, idx] = weights[i]
          edge_index[1, idx] = weights[i+1]

          edge_weights_noise[start+idx] = weights[i+3]
          edge_weights[start+idx] = weights[i+2]
          

          edge_index[1, start+idx] = weights[i]
          edge_index[0, start+idx] = weights[i+1]

      
      for i in range(0, res.shape[0], 2):
          res_box.append((res[i], res[i+1]))
      
      res_set = set(res_box)
      res_cost_sum = 0

      edge_index = torch.from_numpy(edge_index).long()
      
      for i in range(edge_index.shape[1]):
          u, v = edge_index[0, i].item(), edge_index[1, i].item()  
          if (u, v) in res_set or (v,u) in res_set:
              edge_attr[i] = 1  
              res_cost_sum += edge_weights[i]
     
      point_indicator = np.array([label.shape[0]], dtype=np.int64)
      edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)

      graph_data = GraphData(x=torch.from_numpy(label).float(),
                            edge_index=edge_index,
                            edge_attr=torch.from_numpy(edge_attr).float(),
                            edge_weights = torch.from_numpy(edge_weights).float(),
                            edge_weights_noise = torch.from_numpy(edge_weights_noise).float(),
                            res_cost = res_cost_sum/2,
                            point_indicator = point_indicator,
                            edge_indicator = edge_indicator
                            )

      return graph_data
    
"""Lightning module for training the DP-GST model."""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import networkx as nx
from co_datasets.gst_dataset import GSTDataset
from utils.diffusion_schedulers import InferenceSchedule
from pl_meta_model import COMetaModel
import math
 
class GSTModel(COMetaModel):
    def __init__(self,
               param_args=None):
        super(GSTModel, self).__init__(param_args=param_args, node_feature_only=False)

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

       
        self.val_solved_costs = []  # store solved_cost for each validation batch
        # Add a list of pred_cost and true_cost to be printed in the test phase.
        self.pred_costs = []
        self.time = []
        # Record time and error for intermediate steps
        self.process_time = []
        self.process_error = []
        self.val_solved_cost_mean = None

    def forward(self, label, edge_weights_noise, adj_matrix , xt, t,edge_index=None):
        return self.model(label, edge_weights_noise , adj_matrix , xt, t,edge_index)

    def gaussian_training_step(self, batch, batch_idx):
        if self.sparse:
        # TODO: Implement Gaussian diffusion with sparse graphs
            raise ValueError("DIFUSCO with sparse graphs are not supported for Gaussian diffusion")
        
        batch_index,labels,_, edge_weights_noise, adj_matrix, res_matrix , _= batch
        res_matrix = res_matrix * 2 - 1
        res_matrix = res_matrix * (1.0 + 0.05 * torch.rand_like(res_matrix))
        # Sample from diffusion
        t = np.random.randint(1, self.diffusion.T + 1, res_matrix.shape[0]).astype(int)
        xt, epsilon = self.diffusion.sample(res_matrix, t)
       
        t = torch.from_numpy(t).float().view(res_matrix.shape[0])
        # Denoise
        epsilon_pred = self.forward(
            labels.float().to(res_matrix.device),
            edge_weights_noise.float().to(res_matrix.device),
            adj_matrix.float().to(res_matrix.device),
            xt.float().to(res_matrix.device),
            t.float().to(res_matrix.device),
        )
        epsilon_pred = epsilon_pred.squeeze(1)
        loss = F.mse_loss(epsilon_pred, epsilon.float())

        self.log("train/loss", loss)
        return loss


    def categorical_training_step(self, batch, batch_idx):
        edge_index = None
        if not self.sparse:
            batch_index,labels, _,edge_weights_noise, adj_matrix, res_matrix,_ = batch
            t = np.random.randint(1, self.diffusion.T + 1, labels.shape[0]).astype(int)
        else:
            graph_data = batch
            edge_attr = graph_data.edge_attr#Storing Edges in Group Steiner Trees
            labels = graph_data.x
            edge_index = graph_data.edge_index# Denotes an edge from (0, i) to (1, i)
            edge_weights_noise = graph_data.edge_weights_noise
            edge_weights = graph_data.edge_weights
            point_indicator = graph_data.point_indicator
            t = np.random.randint(1, self.diffusion.T + 1, 1).astype(int)
            edge_index,edge_attr,edge_weights,edge_weights_noise = self.change_edge(labels,edge_index,edge_attr,edge_weights,edge_weights_noise)

        res_matrix = edge_attr.reshape((1, -1))
        res_matrix_onehot = F.one_hot(res_matrix.long(), num_classes=2).float()
        if self.sparse:
            res_matrix_onehot = res_matrix_onehot.unsqueeze(1)

        xt = self.diffusion.sample(res_matrix_onehot, t)
        xt = xt * 2 - 1
        xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

        if self.sparse:
            t = torch.from_numpy(t).float()
            t = t.reshape(-1, 1).repeat(1, res_matrix.shape[1]).reshape(-1)
            xt = xt.reshape(-1)
            labels = labels.reshape(-1)
            edge_weights_noise=edge_weights_noise.reshape(-1)
            edge_index = edge_index.float().to(edge_attr.device).reshape(2, -1)
        else:
            t = torch.from_numpy(t).float().view(res_matrix.shape[0])
        # Denoise
        x0_pred = self.forward(
            labels.float().to(edge_attr.device),
            edge_weights_noise.float().to(edge_attr.device),
            None,
            xt.float().to(edge_attr.device),
            t.float().to(edge_attr.device),
            edge_index,
        )
        res_matrix = res_matrix.reshape(-1)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(x0_pred, res_matrix.long())
        self.log("train/loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        if self.diffusion_type == 'gaussian':
            return self.gaussian_training_step(batch, batch_idx)
        elif self.diffusion_type == 'categorical':
            return self.categorical_training_step(batch, batch_idx)

    def gaussian_denoise_step(self, labels,edge_weights_noise,adj_matrix, xt, t, device, target_t=None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            pred = self.forward(
                labels.float().to(device),
                edge_weights_noise.float().to(device),
                adj_matrix.float().to(device),
                xt.float().to(device),
                t.float().to(device),
            )
            pred = pred.squeeze(1)
            xt = self.gaussian_posterior(target_t, t, pred, xt)
            return xt

    def categorical_denoise_step(self, labels,edge_weights_noise,edge_index, xt, t, device, target_t=None):
        with torch.no_grad():
            t = torch.from_numpy(t).view(1)
            x0_pred = self.forward(
                labels.float().to(device),
                edge_weights_noise.float().to(device),
                None,
                xt.float().to(device),
                t.float().to(device),
                edge_index,
            )

            if not self.sparse:
                x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            else:
                x0_pred_prob = x0_pred.reshape((1, labels.shape[0], -1, 2)).softmax(dim=-1)
            xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
            return xt
    
    def change_edge(self,label,edge_index,edge_attr,edge_weights,edge_weights_noise):
      dim_x = label.shape[0]
      if edge_index.shape[1]%dim_x == 0:
        return edge_index,edge_attr,edge_weights,edge_weights_noise
      else:
        add_num = dim_x - edge_index.shape[1]%dim_x
        # 创建自环边
        loop_edges = torch.tensor([[0], [0]], 
                                dtype=edge_index.dtype,
                                device=edge_index.device).repeat(1, add_num)
        
        # 拼接新边
        new_edge_index = torch.cat([edge_index, loop_edges], dim=1)
        # 创建新特征（默认填充0）
        def _pad(arr, add_num):
          return torch.cat([
            arr, 
            torch.zeros(add_num, dtype=arr.dtype, device=arr.device)
            ])
        new_edge_attr = _pad(edge_attr, add_num)
        new_weights = _pad(edge_weights, add_num)
        new_weights_noise = _pad(edge_weights_noise, add_num)
        
        return new_edge_index, new_edge_attr, new_weights, new_weights_noise
    
    
    def test_step(self, batch, batch_idx, split='test'):
        start = time.perf_counter()
        edge_index = None
        if not self.sparse:
            batch_index,labels, edge_weights, edge_weights_noise, adj_matrix, res_matrix,res_cost = batch
        else:
            graph_data = batch
            edge_attr = graph_data.edge_attr
            labels = graph_data.x
            edge_index = graph_data.edge_index
            edge_weights_noise = graph_data.edge_weights_noise
            edge_weights = graph_data.edge_weights
            res_cost = graph_data.res_cost
            edge_index,edge_attr,edge_weights,edge_weights_noise = self.change_edge(labels,edge_index,edge_attr,edge_weights,edge_weights_noise)

        device = edge_attr.device
        min_cost = float('inf')
        res_cost=res_cost.item()
        for sam in range(1,self.args.sequential_sampling+1):

            xt = torch.randn_like(edge_attr.float())

            if self.diffusion_type == 'gaussian':
                xt.requires_grad = True
            else:
                xt = (xt > 0).long()

            if self.sparse:
                xt = xt.reshape(-1)

            steps = self.args.inference_diffusion_steps
            time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                            T=self.diffusion.T, inference_T=steps)

        # Diffusion iterations
            for i in range(steps):
                t1, t2 = time_schedule(i)
                t1 = np.array([t1]).astype(int)
                t2 = np.array([t2]).astype(int)
                if self.diffusion_type == 'gaussian':
                    xt = self.gaussian_denoise_step(
                        labels, edge_weights_noise, adj_matrix, xt,t1,device, target_t=t2)
                else:
                    xt = self.categorical_denoise_step(
                        labels,edge_weights_noise,edge_index, xt, t1, device, target_t=t2)
            if self.diffusion_type == 'gaussian':
                adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
            else:
                adj_mat = xt.float().cpu().detach().numpy() + 1e-6
            
            if not self.sparse  :
                pred_cost=self.find_minimum_spanning_subgraph_dense(adj_mat,adj_matrix,labels,edge_weights)
            else :
                pred_cost=self.find_minimum_spanning_subgraph(adj_mat,edge_index,labels,edge_weights)

            if pred_cost<min_cost:
                min_cost=pred_cost
            # Record multi-step time and min_cost
            if sam==1:
                min_cost1=min_cost
                end1 = time.perf_counter()
                test_time1 = end1 - start
            if sam==4:
                min_cost4=min_cost
                end4 = time.perf_counter()
                test_time4 = end4 - start
            if sam==8:
                min_cost8=min_cost
                end8 = time.perf_counter()
                test_time8 = end8 - start
            if sam==16:
                min_cost16=min_cost
                end16 = time.perf_counter()
                test_time16 = end16 - start

        diff = abs((abs(min_cost)-abs(res_cost))/res_cost)
        print(f"error :{diff}")
        end = time.perf_counter()
        if split == 'test':
            test_time = end - start
            self.time.append(torch.tensor(test_time))
            self.pred_costs.append(torch.tensor([min_cost, res_cost], device=self.device)) 
            metrics = None
            # Record multi-step time and min_cost
            if self.args.sequential_sampling==4:
                self.process_time.append(torch.tensor(test_time1, device=self.device))
                self.process_error.append(torch.tensor(min_cost1, device=self.device))
            elif self.args.sequential_sampling==8:
                self.process_time.append(torch.tensor([test_time1,test_time4], device=self.device))
                self.process_error.append(torch.tensor([min_cost1,min_cost4], device=self.device))
            elif self.args.sequential_sampling==16:
                self.process_time.append(torch.tensor([test_time1,test_time4,test_time8], device=self.device))
                self.process_error.append(torch.tensor([min_cost1,min_cost4,min_cost8], device=self.device))
            elif self.args.sequential_sampling==32:
                self.process_time.append(torch.tensor([test_time1,test_time4,test_time8,test_time16], device=self.device))
                self.process_error.append(torch.tensor([min_cost1,min_cost4,min_cost8,min_cost16], device=self.device))
        else:
            metrics = {
                f"{split}/ans_loss": diff
            }
        self.val_solved_costs.append(torch.tensor(diff))
        return metrics

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split='val')

    def validation_epoch_end(self, outputs):
        # Calculate the average of solved_cost for the entire validation set
        val_solved_cost_mean = torch.mean(torch.stack(self.val_solved_costs))

        if len(self.val_solved_costs) > 0:
            self.log('val_solved_cost_mean', val_solved_cost_mean,prog_bar=True, on_step=False, on_epoch=True,sync_dist=True,reduce_fx='mean')
        metrics = {
            'val_solved_cost_mean': self.trainer.callback_metrics.get('val_solved_cost_mean'),
        }
        print(f'metrics:{metrics}')
        self.val_solved_costs = []  # Clearing the list in preparation for the next validation cycle
    
    def test_epoch_end(self, outputs):
        super().test_epoch_end(outputs)

        dir_path = os.path.dirname(self.args.cost_path)
        os.makedirs(dir_path) if not os.path.exists(dir_path) else print(f"The directory {dir_path} already exists.")

        time_mean_str = ""
        error_mean_str = ""
        device = self.trainer.strategy.root_device
        if self.time:
            time_tensor = torch.stack(self.time).to(device)
            gathered_time_costs = self.trainer.strategy.all_gather(time_tensor)
            # Merge all data (shape [num_processes, num_samples, 2] -> [total_samples, 2])
            if self.trainer.is_global_zero:
                time_mean = torch.mean(gathered_time_costs)
        col = int(math.log(self.args.sequential_sampling,2)-1)
        if self.process_time:
            process_time_tensor = torch.stack(self.process_time).to(device)
            process_time_costs = self.trainer.strategy.all_gather(process_time_tensor)
            if self.trainer.is_global_zero:
                all_process_time_costs = process_time_costs.view(-1,col)
                process_time_mean = torch.mean(all_process_time_costs,dim=0)
                time_mean_str = " ".join([f"{x:.4f}" for x in process_time_mean.cpu().numpy()])
        # Collect pred_costs data for all processes
        if self.pred_costs:
            pred_costs_tensor = torch.stack(self.pred_costs).to(device)
            gathered_pred_costs = self.trainer.strategy.all_gather(pred_costs_tensor)
            if self.trainer.is_global_zero:
                all_pred_costs = gathered_pred_costs.view(-1, 2)
                pred_costs_mean = torch.mean(all_pred_costs,dim=0)
                val_solved_cost_mean = abs(pred_costs_mean[0]-pred_costs_mean[1])/pred_costs_mean[1]
        if self.process_error:
            error_tensor = torch.stack(self.process_error).to(device)
            error_costs = self.trainer.strategy.all_gather(error_tensor)
            if self.trainer.is_global_zero:
                all_error_costs = error_costs.view(-1, col)
                error_mean = torch.mean(all_error_costs,dim=0)
                error_mean_1 = abs(error_mean-pred_costs_mean[1])/pred_costs_mean[1]
                error_mean_1 = error_mean_1.cpu()
                error_mean_str = " ".join([f"{x:.4f}" for x in error_mean_1.numpy()])
        if self.trainer.is_global_zero:
            with open(self.args.cost_path, 'w') as f:
                f.write(f"error\t{val_solved_cost_mean}\n")
                f.write(f"time_mean\t{time_mean}\n")
                f.write("error1\terror4\terror8\terror16\t:\n")
                f.write(error_mean_str + "\n")
                f.write("time1\ttime4\ttime8\ttime16\t:\n")
                f.write(time_mean_str + "\n")
                f.write("pred_mean\ttarget_mean:\n")
                f.write(f"{pred_costs_mean[0]}\t{pred_costs_mean[1]}\n")
                f.write("pred\ttarget:\n")
                for row in all_pred_costs:
                    f.write(f"{row[0]}\t{row[1]}\n")
        
        
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
        #Start traversing from the edge with the highest probability and gradually add it to the graph G until the G-connected component covers all the groups, and then it is generated using the minimum spanning tree.
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

    def find_minimum_spanning_subgraph_dense(self,edge_probs, edge_index, groups,edge_weights):
        """
        Args:
            edge_probs: matrix like (V, V), representing the probabilities of edges
            edge_index: matrix like (V, V), representing the connectivity of edges
            groups: the group to which each node belongs to, like (V,)

        Returns:
            returns the edges in the minimum spanning tree over all groups
        """
        # Step 1: Flatten the edge_probs and edge_index matrices.
        edge_probs, edge_index, groups, edge_weights = self.to_cpu(edge_probs, edge_index, groups, edge_weights)
        edge_probs = np.squeeze(edge_probs)  # edge_probs shape: (V, V)
        edge_probs = edge_probs + edge_probs.T
        edge_index = np.squeeze(edge_index)  # edge_index shape: (V, V)
        groups = np.squeeze(groups)  # groups shape: (V,)
        edge_weights = np.squeeze(edge_weights)  # edge_weights shape: (V, V)
        V = edge_probs.shape[0]  # V is the number of nodes

        flattened_probs = edge_probs[edge_index == 1]  # flattened_probs shape: (E,)
        flattened_weights = edge_weights[edge_index == 1]  # flattened_weights shape: (E,)
        flattened_edges = np.array(np.where(edge_index == 1)).T  # flattened_edges shape: (E, 2)

        # Step 2: order the edges according to their probability from highest to lowest
        ratio = flattened_probs
        sorted_indices = np.argsort(ratio)[::-1].copy()
        sorted_edges = flattened_edges[sorted_indices]
        sorted_weights = flattened_weights[sorted_indices]  # sorted_weights shape: (E,)

        # Step 3: create a graph
        G = nx.Graph()
        unique_groups = np.unique(groups[groups != 0])     # unique_groups shape: (G,)
        
        for i in range(sorted_edges.shape[0]):
            u, v = sorted_edges[i]
            G.add_edge(u, v,  weight=sorted_weights[i].item())
            connected_components = list(nx.connected_components(G))
            for component in connected_components:
                unique_groups_in_component = np.unique(
                    groups[list(component)])
                covered_groups = np.sort(
                    unique_groups_in_component[unique_groups_in_component != 0])

                if covered_groups.shape == unique_groups.shape:
                    # Step 4: create a minimum spanning tree for subgraphs covering all groups
                    subgraph = G.subgraph(component)
                    mst = nx.minimum_spanning_tree(subgraph)

                    # Step 5: trimming a minimum spanning tree
                    if self._covers_all_groups(mst, groups, unique_groups):
                        return self._prune_tree(mst, groups, unique_groups)

        return None

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

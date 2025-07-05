"""Generating noisy datasets"""

import argparse
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gam", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--num_train", type=int, default=1600)
    parser.add_argument("--num_test_val", type=int, default=40)
    parser.add_argument("--num_query_task", type=int, default=30)
    parser.add_argument("--dataname", type=str, default=None)
    parser.add_argument("--source_file", type=str, default=None)
    parser.add_argument("--target_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    opts = parser.parse_args()

    np.random.seed(opts.seed)
    gam=opts.gam
    name = opts.dataname
    epsilon = opts.epsilon
    delta = opts.delta
    n = 1
    if opts.source_file==None:
        data_file = f'./Data_for_GNN_new/g{gam}/{name}_data_g{gam}_1k.txt'
    else:
        data_file = opts.source_file
    if opts.target_file == None:
        file_path = f"./Data_for_GNN_noise/tasks_30-num_2000-epsilon_{epsilon}-delta_{delta}/{name}_GST1k-gamma_{gam}-noise"
    else:
        file_path = opts.target_file
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    def find_split_line(data_file,n):
        with open(data_file, 'r') as file:
            file_lines=[]
            i=(n-1)*20
            for line in file.readlines():

                line1 = line.strip()
                file_lines.append(line1)
        start_index = (n - 1) * 20
        end_index = start_index + 20
        return file_lines[start_index:end_index]

    file_lines_1 = find_split_line(data_file,1)
    file_lines_2 = find_split_line(data_file,2)
    file_lines_3 = find_split_line(data_file,3)
    file_lines_4 = find_split_line(data_file,4)
    file_lines_5 = find_split_line(data_file,5)
    file_lines_6 = find_split_line(data_file,6)
    def get_example(idx):
        try:
            # Get the line and strip leading/trailing whitespace
            if n%6==0:
                line = file_lines_2[idx].strip()
            elif n % 6 == 1:
                line = file_lines_1[idx].strip()
            elif n % 6 == 2:
                line = file_lines_3[idx].strip()
            elif n % 6 == 3:
                line = file_lines_4[idx].strip()
            elif n % 6 == 4:
                line = file_lines_5[idx].strip()
            elif n % 6 == 5:
                line = file_lines_6[idx].strip()
            # label information that stores which group the vertex belongs to
            label_line = line.split('Label ')[1]
            label_line2 = label_line.split('|GroupNum')[0]
            label = np.array([int(t) for t in label_line2.split(' ') if t != ''])

            parts = line.split("|")
            # Get the part with index 1, i.e. |GroupNum ... Query ... Query ... |Query ...
            other = parts[1]

            # Weight information, which stores the weights between edges
            graph = label_line.split('Graph ')[1]
            graph2 = graph.split('Result ')[0]
            weights = np.array([int(t) for t in graph2.split(' ') if t != ''])

            # Results for Storage Group Steiner Tree
            tree = graph.split('Result ')[1]
            res = np.array([int(t) for t in tree.split(' ') if t != ''])
            return label, weights, res, other

        except (IndexError, ValueError) as e:
            print(f"Error parsing line at index {idx}: {e}")
            return None


    def getdata_noise(idx,st=None):

        label, weights, res,other =get_example(idx)
        data_lines = []
        edge_weights = np.zeros(weights.shape[0] // 3)

        for idx, i in enumerate(range(0, weights.shape[0], 3)):
            edge_weights[idx] = weights[i + 2]
        if st =="train":
            num_data = opts.num_train
        else:
            num_data = opts.num_test_val
        for j in range(num_data):
            edge_weights_noise = add_noise_to_edge_weights(edge_weights, epsilon, delta)
            new_list = []
            index = 0
            i = 0
            while index < len(weights):
                new_list.append(weights[index])
                if (index + 1) % 3 == 0:
                    new_list.append(int(edge_weights_noise[i]))
                    i+=1
                index += 1
            print(f"j:{j}")
            data_line = "Label"+" "+" ".join(map(str,label)) + "|" + other + "|" +"Graph"+" "+" ".join(map(str, new_list)) + " " + "Result"+" "+" ".join(map(str, res))
            data_lines.append(data_line)

        if st =="train":
            return data_lines
        else:
            val_data_temp, test_data_temp = train_test_split(data_lines, test_size=0.5, random_state=42)
            return test_data_temp, val_data_temp


    def add_noise_to_edge_weights(edge_weights, epsilon, delta):
        """
        Add the same privacy noise to non-zero values in edge_weights and add an offset term.

        This function can handle one-dimensional arrays or adjacency matrices (two-dimensional arrays).

        Args.
        - edge_weights: 1D or 2D numpy array, containing the weights of the edges or the adjacency matrix.
        - epsilon: privacy noise parameter that determines the magnitude of the noise.
        - delta: parameter controlling the probability of failure.

        Returns.
        - noisy_edge_weights: numpy array with the same dimensions as the input, plus edge weights offset by noise and delta.
        """

        if edge_weights.ndim == 1:
            E = np.count_nonzero(edge_weights)
        elif edge_weights.ndim == 2:
            E = np.count_nonzero(edge_weights)
        else:
            raise ValueError("Edge_weights can only be one or two dimensional arrays")

        delta = (1 / epsilon) * np.log(E / delta)

        # noise = np.random.laplace(1 / epsilon)

        if edge_weights.ndim == 1:
            noisy_edge_weights = np.array([w + np.random.laplace(1 / epsilon) + delta if w != 0 else w for w in edge_weights])


        elif edge_weights.ndim == 2:
            noisy_edge_weights = np.copy(edge_weights)

            rows, cols = edge_weights.shape
            for i in range(rows):
                for j in range(cols):
                    if edge_weights[i, j] != 0:
                        noisy_edge_weights[i, j] += np.random.laplace(1 / epsilon) + delta

        return noisy_edge_weights

    def getdata_file(num_graph):# Num_graph denotes the number of graphs for different query tasks
        global n
        train_data = []
        test_data = []
        val_data = []
        for _ in range(1,num_graph+1):
            print(f"Start {n} query tasks:")
            st = "train"
            if n%6==0:
                num = random.randint(0,len(file_lines_2)-1)
                train_data_temp=getdata_noise(num,st)
                train_data.extend(train_data_temp)
                test_data_temp, val_data_temp = getdata_noise(num)
                test_data.extend(test_data_temp)
                val_data.extend(val_data_temp)
                del file_lines_2[num]
            elif n%6==1:
                num = random.randint(0, len(file_lines_1) - 1)
                train_data_temp = getdata_noise(num, st)
                train_data.extend(train_data_temp)
                test_data_temp, val_data_temp = getdata_noise(num)
                test_data.extend(test_data_temp)
                val_data.extend(val_data_temp)
                del file_lines_1[num]
            elif n%6==2:
                num = random.randint(0, len(file_lines_3) - 1)
                train_data_temp = getdata_noise(num, st)
                train_data.extend(train_data_temp)
                test_data_temp, val_data_temp = getdata_noise(num)
                test_data.extend(test_data_temp)
                val_data.extend(val_data_temp)
                del file_lines_3[num]
            elif n%6==3:
                num = random.randint(0, len(file_lines_4) - 1)
                train_data_temp = getdata_noise(num, st)
                train_data.extend(train_data_temp)
                test_data_temp, val_data_temp = getdata_noise(num)
                test_data.extend(test_data_temp)
                val_data.extend(val_data_temp)
                del file_lines_4[num]
            elif n%6==4:
                num = random.randint(0, len(file_lines_5) - 1)
                train_data_temp = getdata_noise(num, st)
                train_data.extend(train_data_temp)
                test_data_temp, val_data_temp = getdata_noise(num)
                test_data.extend(test_data_temp)
                val_data.extend(val_data_temp)
                del file_lines_5[num]
            elif n%6==5:
                num = random.randint(0, len(file_lines_6) - 1)
                train_data_temp = getdata_noise(num, st)
                train_data.extend(train_data_temp)
                test_data_temp, val_data_temp = getdata_noise(num)
                test_data.extend(test_data_temp)
                val_data.extend(val_data_temp)
                del file_lines_6[num]
            if n % 6 == 0:
                random.shuffle(train_data)
                random.shuffle(test_data)
                random.shuffle(val_data)
                test_data_120 = test_data[:120]
                val_data_120 = val_data[:120]
                with open(
                        os.path.join(file_path, 'train_split.txt'),
                        'a') as f:
                    for line in train_data:
                        f.write(line + "\n")

                with open(
                        os.path.join(file_path, 'valid_split.txt'),
                        'a') as f:
                    for line in val_data_120:
                        f.write(line + "\n")

                with open(
                        os.path.join(file_path, 'test_split.txt'),
                        'a') as f:
                    for line in test_data_120:
                        f.write(line + "\n")
                train_data = []
                test_data = []
                val_data = []
            print(f"Finish {n} query tasks:")
            n = n+1


    getdata_file(opts.num_query_task)
    print(f"")
    print(f"Noise additions for {opts.num_query_task} query tasks completed")

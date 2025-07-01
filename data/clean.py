
import numpy as np
import os

def parse_graph_data(line):
    """Parses the Graph portion of the original data row and extracts all edges"""
    try:
        label_line = line.split('Label ')[1]
        graph = label_line.split('Graph ')[1]
        graph2 = graph.split('Result ')[0]
        numbers = np.array([int(t) for t in graph2.split(' ') if t != ''])
    except StopIteration:
        print("Warning: Graph section not found")
        return []

    # Extract all numbers and group them into triples
    edges = []
    for i in range(0, len(numbers), 3):
        if i + 2 < len(numbers):
            u, v, w = numbers[i], numbers[i + 1], numbers[i + 2]
            edges.append((u, v, w))
    return edges


def remove_duplicate_bidirectional_edges(edges):
    """Remove duplicates from bi-directional edges"""
    seen = set()
    cleaned = []
    for u, v, w in edges:
        # Generate standardized keys (ignore direction)
        key = tuple(sorted([u, v]))
        if key not in seen:
            cleaned.append((u, v, w))
            seen.add(key)
    return cleaned


def rebuild_data_line(line, cleaned_edges):
    """Rebuild complete data rows"""
    # Split the original row parts
    label_line = line.split('Label ')[1]
    label_line2 = label_line.split('|GroupNum')[0]
    label = np.array([int(t) for t in label_line2.split(' ') if t != ''])
    parts = line.split("|")
    other = parts[1]

    graph = label_line.split('Graph ')[1]
    graph2 = graph.split('Result ')[0]
    weights = np.array([int(t) for t in graph2.split(' ') if t != ''])

    tree = graph.split('Result ')[1]
    res = np.array([int(t) for t in tree.split(' ') if t != ''])
    # Generate a new Graph section
    new_graph = "Graph " + " ".join([f"{u} {v} {w}" for u, v, w in cleaned_edges])

    data_line = "Label" + " " + " ".join(map(str, label)) + "|" + other + "|" + new_graph+" " + "Result" + " " + " ".join(map(str, res))
    return data_line

def safe_path(path):
    return os.path.abspath(os.path.expanduser(path)).replace("\\", "/")

def process_file(input_file):
    new_lines = []
    """Batch processing of documents"""
    with open(input_file, 'r') as fin:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            # 处理流程
            try:
                edges = parse_graph_data(line)
                cleaned_edges = remove_duplicate_bidirectional_edges(edges)
                new_line = rebuild_data_line(line, cleaned_edges)
                new_lines.append(new_line)
            except Exception as e:
                print(f"Error processing line {idx + 1}. {str(e)}")
                continue
    return new_lines

if __name__ == "__main__":
    # # Processing Folders
    # input_dirs = 'F:/PycharmProjects/GNN/data/Data_for_GNN'
    # output_root = 'F:/PycharmProjects/GNN/data/Data_for_GNN_new'
    # # Iterate through all subfolders in the source directory
    # for folder2 in os.listdir(input_dirs):
    #     folder2_path = os.path.join(input_dirs, folder2)
    #
    #     if not os.path.isdir(folder2_path):
    #         continue
    #
    #     output_subdir = os.path.join(output_root, folder2)
    #     os.makedirs(output_subdir, exist_ok=True)
    #
    #     for filename in os.listdir(folder2_path):
    #         if filename.endswith('.txt'):
    #             input_path = os.path.join(folder2_path, filename)
    #             output_path = os.path.join(output_subdir, filename)
    #             input_path = safe_path(input_path)
    #             output_path = safe_path(output_path)
    #             new_lines = process_file(input_path)
    #             with open(
    #                     output_path,
    #                     'w') as f:
    #                 for line in new_lines:
    #                     f.write(line + "\n")
    #             print(f"Successfully processed. {input_path} -> {output_path}")

    #处理单个文件
    gam = 3
    name = 'youtu'
    input_path = f"F:\PycharmProjects\GNN\data\Data_for_GNN\g{gam}\\{name}_data_g{gam}_1k.txt"
    output_path = f"F:\PycharmProjects\GNN\data\Data_for_GNN_new\g{gam}\\{name}_data_g{gam}_1k.txt"
    new_lines = process_file(input_path)
    with open(
            output_path,
            'w') as f:
        for line in new_lines:
            f.write(line + "\n")
    print(f"Successfully processed. {input_path} -> {output_path}")
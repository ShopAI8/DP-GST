#include "rucgraph/GST_data.hpp"
#include <iostream>
#include <cmath>
#include <random>
using namespace std;

// 生成拉普拉斯噪声
double laplace_noise(double mu, double b, std::mt19937 &gen)
{
    std::uniform_real_distribution<> dis(-0.5, 0.5);          // 均匀分布，[-0.5, 0.5]区间
    return mu - b * std::log(1.0 - 2.0 * std::abs(dis(gen))); // 通过均匀分布生成拉普拉斯噪声
}

double get_noise_for_instance_graph(double epsilon, double gamma, double E)
{
    double delta = (1 / epsilon) * log(E / gamma);

    std::random_device rd;
    std::mt19937 gen(rd());   // 随机数生成器
    double b = 1.0 / epsilon; // 拉普拉斯噪声的尺度参数

    // 生成拉普拉斯噪声并加上delta
    return laplace_noise(0.0, b, gen) + delta;
}

double cal_exp_w(double w_noisy, int E, double epsilon, double gamma)
{
    double w_bar = w_noisy - (1 / epsilon) * log(E / gamma);
    double C = 1 / epsilon * exp(-epsilon * w_bar) - (100 + 1 / epsilon) * exp(-epsilon * (100 - w_bar));
    return w_noisy + C;
}

pair<graph_hash_of_mixed_weighted, graph_hash_of_mixed_weighted> prepare_data(int G, int g_size_min, int g_size_max, double V, double E, double nw_min, double nw_max,
                                                                              double ec_min, double ec_max, int input_precision, unordered_map<int, int> &label, bool &find)
{
    graph_hash_of_mixed_weighted instance_graph = graph_hash_of_mixed_weighted_generate_random_connected_graph(V, E, nw_min, nw_max, ec_min, ec_max, input_precision);
    graph_hash_of_mixed_weighted generated_group_graph;

    std::unordered_set<int> generated_group_vertices;
    graph_hash_of_mixed_weighted_generate_random_groups_of_vertices(G, g_size_min, g_size_max, instance_graph, instance_graph.hash_of_vectors.size(), generated_group_vertices, generated_group_graph, label);

    // cout << "generate complete" << endl;
    unordered_map<int, int> vertexID_old_to_new;
    for (int mm = 0; mm < V; mm++)
    {
        vertexID_old_to_new[mm] = mm;
    }
    graph_v_of_v_idealID v_instance_graph = graph_hash_of_mixed_weighted_to_graph_v_of_v_idealID(instance_graph, vertexID_old_to_new);
    for (int mm = V; mm < V + G; mm++)
    {
        vertexID_old_to_new[mm] = mm;
    }
    // CSR_graph csr_graph = toCSR(v_instance_graph);
    // cout << "E:" << csr_graph.E_all << " v:" << csr_graph.V << endl;
    graph_v_of_v_idealID v_generated_group_graph = graph_hash_of_mixed_weighted_to_graph_v_of_v_idealID(generated_group_graph, vertexID_old_to_new);
    int RAM;
    int p_cpu = 0;
    int *pointer = &p_cpu;
    graph_hash_of_mixed_weighted solu = graph_v_of_v_idealID_PrunedDPPlusPlus(v_instance_graph, v_generated_group_graph, generated_group_vertices, 1, RAM, pointer, find);

    return make_pair(instance_graph, solu);
}

std::tuple<graph_hash_of_mixed_weighted, graph_hash_of_mixed_weighted, graph_hash_of_mixed_weighted, graph_hash_of_mixed_weighted> prepare_data_depend_on_instance_graph(int G, int g_size_min, int g_size_max, double V, double E, double nw_min, double nw_max,
                                                                                                                                                                         double ec_min, double ec_max, int input_precision, unordered_map<int, int> &label, bool &find, bool &find_noise, graph_hash_of_mixed_weighted instance_graph)
{
    // graph_hash_of_mixed_weighted instance_graph = graph_hash_of_mixed_weighted_generate_random_connected_graph(V,E,nw_min,nw_max,ec_min,ec_max,input_precision);
    graph_hash_of_mixed_weighted generated_group_graph;

    std::unordered_set<int> generated_group_vertices;
    graph_hash_of_mixed_weighted_generate_random_groups_of_vertices(G, g_size_min, g_size_max, instance_graph, instance_graph.hash_of_vectors.size(), generated_group_vertices, generated_group_graph, label);

    graph_hash_of_mixed_weighted instance_graph_noise = instance_graph;
    for (int i = 0; i < instance_graph_noise.hash_of_vectors.size(); i++)
    {
        for (int j = 0; j < instance_graph_noise.hash_of_vectors[i].adj_vertices.size(); j++)
        {
            instance_graph_noise.hash_of_vectors[i].adj_vertices[j].second += get_noise_for_instance_graph(2, 0.05, E);
        }
    }
    // cout << "generate complete" << endl;
    unordered_map<int, int> vertexID_old_to_new;
    for (int mm = 0; mm < V; mm++)
    {
        vertexID_old_to_new[mm] = mm;
    }
    graph_v_of_v_idealID v_instance_graph_noise = graph_hash_of_mixed_weighted_to_graph_v_of_v_idealID(instance_graph_noise, vertexID_old_to_new);
    graph_v_of_v_idealID v_instance_graph = graph_hash_of_mixed_weighted_to_graph_v_of_v_idealID(instance_graph, vertexID_old_to_new);
    for (int mm = V; mm < V + G; mm++)
    {
        vertexID_old_to_new[mm] = mm;
    }
    // CSR_graph csr_graph = toCSR(v_instance_graph);
    // cout << "E:" << csr_graph.E_all << " v:" << csr_graph.V << endl;
    graph_v_of_v_idealID v_generated_group_graph = graph_hash_of_mixed_weighted_to_graph_v_of_v_idealID(generated_group_graph, vertexID_old_to_new);
    int RAM;
    int p_cpu = 0;
    int *pointer = &p_cpu;
    graph_hash_of_mixed_weighted solu = graph_v_of_v_idealID_PrunedDPPlusPlus(v_instance_graph, v_generated_group_graph, generated_group_vertices, 1, RAM, pointer, find);
    graph_hash_of_mixed_weighted solu_noise = graph_v_of_v_idealID_PrunedDPPlusPlus(v_instance_graph_noise, v_generated_group_graph, generated_group_vertices, 1, RAM, pointer, find_noise);
    return std::make_tuple(instance_graph, instance_graph_noise, solu, solu_noise);
}

double cal_cost(graph_v_of_v_idealID &instance_graph, graph_hash_of_mixed_weighted &ans)
{
    double cost = 0;
    for (auto it : ans.hash_of_vectors)
    {
        for (auto its : it.second.adj_vertices)
        {
            // file<<it.first<<" "<<its.first<<" ";
            auto a = instance_graph[it.first];
            for (int i = 0; i < a.size(); i++)
            {
                if (a[i].first == its.first)
                {
                    cost += a[i].second;
                }
            }
        }
    }
    return cost / 2;
}
// 获取原图中权重的最大值
std::pair<double, int> find_max_weight(graph_v_of_v_idealID &instance_graph)
{
    double max = 0;
    int count = 0;
    for (auto v : instance_graph)
    {
        for (auto t : v)
        {
            if (max < t.second)
                max = t.second;
            count++;
        }
    }
    return {max, count / 2};
}
// 计算期望权重
void adjust_noise_weights(double epsilon, double gam, graph_v_of_v_idealID &v_sub_graph, graph_v_of_v_idealID &v_sub_graph_noise, graph_v_of_v_idealID &v_sub_generated_group_graph_noise)
{
    auto [m, E] = find_max_weight(v_sub_graph);

    for (int u = 0; u < v_sub_graph_noise.size(); ++u)
    {
        for (int v = 0; v < v_sub_graph_noise[u].size(); ++v)
        {
            auto t = v_sub_graph_noise[u][v];
            double w = t.second;
            double delta = (1.0 / epsilon) * std::log(E / gam);
            double w_bar = w - delta;
            double e_w=0;
            if (w_bar >= m)
            {
                e_w = (w_bar - 1 / epsilon)/ 2 + std::exp(-epsilon * w_bar) / (2 * epsilon);
                // e_w = ((m - 1 / epsilon)*std::exp(epsilon * (m - w + delta)))/ 2 + std::exp(-epsilon * w_bar) / (2 * epsilon);
            }
            else
            {
                double term1 = std::exp(-epsilon * (w - delta)) / epsilon;
                double term2 = (m + 1.0 / epsilon) * std::exp(-epsilon * (m - w + delta));
                e_w = w + (term1 - term2) / 2 - delta;
            }
            v_sub_graph_noise[u][v].second = e_w;
            v_sub_generated_group_graph_noise[u][v].second = e_w;
            if (e_w < 0)
            // if(w_bar>=m)
                cout << w << "  " << e_w << "  "<< delta << endl;
        }
    }
}
// void read_graph_from_file(const string& filename){
//     ifstream file(filename);
//     string line;

//     if (!file.is_open()) {
//         cerr << "Unable to open file!" << endl;
//         return;
//     }

//     while (getline(file, line)) {
//         stringstream ss(line);
//         string word;
//         unordered_map<int, int> labels;
//         // 读取 Labels
//         ss >> word; // "Label"
//         int cnt=0;
//         while (ss >> word && word != "Graph") {
//             int label = std::stoi(word);  // 将标签从字符串转换为整数
//             labels[cnt++] = label;   // 假设标签就是其自身的 ID
//         }

//         // 读取 Graph
//         // ss >> word; // "Graph"
//         int u, v;
//         double weight, noise_weight;
//         while (ss >> node >> adj_node >> weight >> noise_weight) {
//             graph.hash_of_vectors[node].push_back({adj_node, weight});
//             graph_noise.hash_of_vectors[node].push_back({adj_node, noise_weight});
//         }

//         // 读取 Result
//         ss >> word; // "Result"
//         while (ss >> node >> adj_node) {
//             // 假设这是计算结果，这里可以进行进一步的处理
//             // 这里只是简单读取，如果需要处理计算结果可以扩展
//         }
//     }

//     file.close();
// }

bool bfs_subgraph(int start, int target_size, vector<int> &subgraph, graph_v_of_v_idealID &v_instance_graph)
{
    if (v_instance_graph.empty())
        return false;
    subgraph.clear();
    unordered_set<int> visited;
    queue<int> q;

    q.push(start);
    visited.insert(start);
    subgraph.push_back(start);

    while (!q.empty() && subgraph.size() < target_size)
    {
        int current = q.front();
        q.pop();

        for (auto it : v_instance_graph[current])
        {
            int neighbor = it.first;
            if (visited.find(neighbor) == visited.end())
            {
                visited.insert(neighbor);
                subgraph.push_back(neighbor);

                q.push(neighbor);

                // 达到目标大小立即停止
                if (subgraph.size() == target_size)
                    return true;
            }
        }
    }

    return false;
}
std::vector<std::string> get_all_files(const std::string &folder_path, bool recursive = false)
{
    std::vector<std::string> file_names;
    try
    {
        if (recursive)
        {
            // 递归遍历所有子目录
            for (const auto &entry : fs::recursive_directory_iterator(folder_path))
            {
                if (entry.is_regular_file())
                { // 仅限普通文件
                    file_names.push_back(entry.path().string());
                }
            }
        }
        else
        {
            // 仅当前目录
            for (const auto &entry : fs::directory_iterator(folder_path))
            {
                if (entry.is_regular_file())
                { // 仅限普通文件
                    file_names.push_back(entry.path().string());
                }
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "错误: " << e.what() << std::endl;
    }
    return file_names;
}
int main(int argc, char *argv[])
{
    cout << "start.." << endl;
    // int Num = atoi(argv[1]);
    // int iteration = atoi(argv[2]);
    // double epsilon = std::stod(argv[3]);
    // double gamma = std::stod(argv[4]);
    int target_size = 1000;
    // string dataset[6]={"twitch","Github","musae","orkut","youtu","dblp"};
    string dataset = "orkut";
    double epsilon = 1; //1,0.5
    double gam = 0.05; //1e-05，0.05
    double total_drop = 0;
    int Num = 0;
    // vector<vector<string>> dataName(2);
    // dataName[0] = get_all_files("../tasks_30-num_2000-epsilon_1-delta_0.05/g3/");
    // dataName[1] = get_all_files("../tasks_30-num_2000-epsilon_1-delta_0.05/g5/");
    // string dataName = "../Data_for_DP2/tasks_30-num_2000-epsilon_0.5-delta_0.05/dblp_GST1k-gamma_3-noise_test.txt";
    string dataName = "/home/sunyahui/DIFUSCO/data/tasks_30-num_2000-epsilon_1-delta_0.05/"+dataset+"_GST1k-gamma_5-noise/test_split.txt";
    // 定义路径对象
    std::filesystem::path dir_path("../result/"+dataset+"-g5-PrunedDP-epsilon1-delta0.05");
    
    // 创建目录（包括所有不存在的父目录）
    std::filesystem::create_directories(dir_path);
    ofstream result(dir_path / "PrunedDP1_result.txt");
    // result << "tasks_30-num_2000-epsilon_1-delta_0.05 result : " << endl;
    result << dataset+"-g5-PrunedDP1-epsilon1-delta0.05 result : " << endl;

    // for(int i =0;i<6;i++){
    // string data_name = dataset[i];
    // string filename;
    // if(Num==3) filename = "../Data_for_Baseline/g3/"+data_name+"_Baseline_g3_1k.txt";
    // if(Num==5) filename = "../Data_for_Baseline/g5/"+data_name+"_Baseline_g5_1k.txt";
    // string filename = "../Github_GST1k-gamma_3-noise.txt";
    for (int gammaNum = 1; gammaNum < 2; gammaNum++)
    {
        if (gammaNum == 0)
        {
            result << "gamma = 3 :" << endl;
            Num = 3;
            gammaNum++;
        }
        else
        {
            result << "gamma = 5 :" << endl;
            Num = 5;
        }
        for (int dataNum = 0; dataNum < 1; dataNum++)
        {
            result << "pred  target : " << endl;
            double total_duration = 0;     // 总耗时累计
            double total_length = 0;       // 原始路径长度累计
            double total_length_noise = 0; // 噪声路径长度累计

            // string filename = dataName[gammaNum][dataNum];

            string filename = dataName;

            std::ifstream myfile(filename);
            string linecontent;
            int count_line = 0;
            if (myfile.is_open()) // if the file is opened successfully
            {
                // cout<<1<<endl;

                int line = 0;
                string tmp;
                double sum_cost = 0;
                double drop = 0;
                int count = 0;
                int E = 0;

                while (getline(myfile, linecontent)) // read file line by line
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    // cout<<2<<endl;
                    line++;
                    graph_v_of_v_idealID v_sub_graph, v_sub_generated_group_graph, v_sub_generated_group_graph_noise, v_sub_graph_noise;
                    vector<pair<int, int>> ans;
                    unordered_set<int> query_group_vertex;
                    vector<int> v_to_g;
                    // set<int> group;
                    size_t split_pos = linecontent.find('|');
                    string result1 = linecontent.substr(split_pos + 1);
                    std::istringstream inputStream(result1);

                    string t;
                    int GroupNum;

                    while (inputStream >> tmp)
                    {
                        if (!tmp.compare("GroupNum"))
                        {
                            // cout<<3<<endl;
                            inputStream >> t;
                            // cout<<t<<endl;
                            GroupNum = std::stoi(t);
                            v_sub_graph.resize(target_size);
                            v_sub_graph_noise.resize(target_size);
                            v_sub_generated_group_graph.resize(target_size + GroupNum);
                            v_sub_generated_group_graph_noise.resize(target_size + GroupNum);
                        }
                        else if (!tmp.compare("Query"))
                        {
                            // cout<<4<<endl;
                            for (int i = 0; i < Num; ++i)
                            {
                                inputStream >> t;
                                query_group_vertex.insert(target_size + std::stoi(t));
                            }
                        }
                        else if (!tmp.compare("Group"))
                        {
                            // cout<<5<<endl;
                            for (int i = 0; i < target_size; ++i)
                            {
                                inputStream >> t;
                                int g = std::stoi(t);
                                v_to_g.push_back(g);
                                // group.insert(g);
                                v_sub_generated_group_graph[i].push_back({target_size + g, 1});
                                v_sub_generated_group_graph[target_size + g].push_back({i, 1});
                                v_sub_generated_group_graph_noise[i].push_back({target_size + g, 1});
                                v_sub_generated_group_graph_noise[target_size + g].push_back({i, 1});
                            }
                        }
                        else if (!tmp.compare("|Graph"))
                        {
                            // cout<<6<<endl;
                            while (inputStream >> t && t.compare("Result"))
                            {
                                int u = std::stoi(t);
                                inputStream >> t;
                                int v = std::stoi(t);
                                inputStream >> t;
                                int w = std::stoi(t);
                                inputStream >> t;
                                int w_noise = std::stoi(t);
                                E++;
                                v_sub_graph[u].push_back({v, w});
                                v_sub_graph[v].push_back({u, w});
                                v_sub_graph_noise[u].push_back({v, w_noise});
                                v_sub_graph_noise[v].push_back({u, w_noise});
                                v_sub_generated_group_graph[u].push_back({v, w});
                                v_sub_generated_group_graph[v].push_back({u, w});
                                v_sub_generated_group_graph_noise[u].push_back({v, w_noise});
                                v_sub_generated_group_graph_noise[v].push_back({u, w_noise});
                            }
                            // cout<<7<<endl;
                            while (inputStream >> t)
                            {
                                int u = std::stoi(t);
                                inputStream >> t;
                                int v = std::stoi(t);
                                ans.push_back({u, v});
                            }
                        }
                    }

                    
                    // std::cin.get(); 
                    double cost = 0;
                    double cost_noise = 0;
                    // cout<<8<<endl;
                    for (auto it : ans)
                    {
                        auto a = v_sub_graph[it.first];
                        for (int i = 0; i < a.size(); i++)
                        {
                            if (a[i].first == it.second)
                            {
                                cost += a[i].second;
                            }
                        }
                    }
                    cost /= 2;
                    total_length += cost;
                    // adjust_noise_weights(epsilon, gam, v_sub_graph, v_sub_graph_noise, v_sub_generated_group_graph_noise);
                    for (size_t i = 0; i < v_sub_graph.size(); i++)
                    {
                        std::sort(v_sub_graph[i].begin(), v_sub_graph[i].end());
                        std::sort(v_sub_graph_noise[i].begin(), v_sub_graph_noise[i].end());
                    }
                    for (size_t i = 0; i < v_sub_generated_group_graph.size(); i++)
                    {
                        std::sort(v_sub_generated_group_graph[i].begin(), v_sub_generated_group_graph[i].end());
                        std::sort(v_sub_generated_group_graph_noise[i].begin(), v_sub_generated_group_graph_noise[i].end());
                    }
                    // int cnt=0;
                    // for(int it =0;it < iteration;it++){
                    // graph_v_of_v_idealID v_sub_graph_noise,v_sub_generated_group_graph_noise;
                    // v_sub_graph_noise=v_sub_graph;
                    // v_sub_generated_group_graph_noise=v_sub_generated_group_graph;
                    // for(int i=0;i < v_sub_graph_noise.size();i++){
                    //     for(int j =0 ;j<v_sub_graph_noise[i].size();j++){
                    //         // int noise = 0;
                    //         int noise = get_noise_for_instance_graph(epsilon,gamma,E/2);
                    //         v_sub_graph_noise[i][j].second=max(noise+v_sub_graph_noise[i][j].second,0.1);
                    //         v_sub_generated_group_graph_noise[i][j].second=max(noise+v_sub_generated_group_graph_noise[i][j].second,0.1);
                    //     }
                    // }
                    // cout<<9<<endl;
                    int RAM;
                    int p_cpu = 0;
                    int *pointer = &p_cpu;
                    bool find = true;
                    graph_hash_of_mixed_weighted solu = graph_v_of_v_idealID_PrunedDPPlusPlus(v_sub_graph_noise, v_sub_generated_group_graph_noise, query_group_vertex, 1, RAM, pointer, find);
                    if (!find)
                    {
                        cout << "   line: " << line << " failed " << endl;
                        continue;
                    }
                    // cout << 10 << endl;
                    // cnt++;
                    count_line++;
                    cost_noise = cal_cost(v_sub_graph, solu);
                    result << cost_noise << "   " << cost << endl;
                    double d = fabs((cost - cost_noise) / cost);
                    
                    total_length_noise += cost_noise;
                    // drop += d;
                    sum_cost += cost_noise;
                    auto end = std::chrono::high_resolution_clock::now();

                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    cout << filename << "   line: " << line << " drop = " << d * 100 << " \% " << "cost " << duration.count() / 1000 << " ms" << endl;
                    total_duration += duration.count() / 1000;
                    // }
                    // count+=cnt;
                    // cout<<"task "<<data_name<<" "<<line++<<" finish with iteration "<<cnt<<endl;
                }
                total_drop = fabs((total_length / 600 - total_length_noise / count_line) / (total_length / 600));
                cout << filename << " finish " << count_line << "tasks   average drop = " << total_drop * 100 << " % with time " << total_duration << " ms" << endl;
                // cout<<data_name<<"'s average cost = "<<sum_cost/count<<" drop = "<<drop/count*100<<"%"<<endl;
                // }
            }
            result << filename << count_line << " tasks  drop = " << total_drop * 100 << " \% time = " << total_duration << "ms average_time = " << total_duration / count_line << "ms  length_actural = " << total_length / 600 << "   length_noise = " << total_length_noise / count_line << endl;
        }
        // cout<<"total average drop of 6 dataset : "<<total_drop/6*100<<"%"<<endl;
    }
    result.close();

    // // 子图生成
    // cout << "start.." << endl;     // 打印启动信息
    // int target_size = 1000;        // 定义目标子图大小为1000个节点

    // // 从命令行参数获取输入参数
    // // int sub_num = atoi(argv[1]);   // 第一个参数：生成子图的数量
    // // int T = atoi(argv[2]);         // 第二个参数：每个子图的查询任务数量
    // // int gamma = atoi(argv[3]);     // 第三个参数：每个查询任务包含的组数
    // int sub_num = 6;   // 第一个参数：生成子图的数量
    // int T = 20;         // 第二个参数：每个子图的查询任务数量
    // int gamma = 5;     // 第三个参数：每个查询任务包含的组数

    // // 预定义数据集名称数组（实际只用到索引5的dblp）
    // string dataset[6]={"twitch","Github","musae","orkut","youtu","dblp"};

    // // 外层循环（设计执行1次，可能用于扩展多数据集处理）
    // for(int j=0;j<1;++j){
    //     string data_name = dataset[1];
    //     // string data_name = "com-amazon";
    //     // string data_name = "com-amazon";

    //     string data_path="";  // 数据文件路径初始化

    //     // 根据gamma值构建不同数据文件路径
    //     if(gamma==3)
    //         data_path = "../Data_merge/g3_new/"+data_name+"_data_g3_1k.txt";  // gamma=3的路径
    //     else if(gamma==5)
    //         data_path = "../Data_merge/g5_new/"+data_name+"_data_g5_1k.txt";  // gamma=5的路径
    //     else return 0;  // 非法gamma值直接退出

    //     ofstream data_file(data_path);  // 创建输出文件流
    //     cout<<"-----------"<<data_name<<"-----------"<<endl;  // 打印当前数据集分隔线

    //     // 声明图结构：原始图实例和分组图结构
    //     graph_v_of_v_idealID group_graph, v_instance_graph;
    //     string path = "/home/sunyahui/lijiayu/GST/data/";  // 数据存储根路径
    //     string path1 = "/home/sunyahui/DIFUSCO/data/graph_data/Github.in";  // in数据存储根路径
    //     // 读取输入图文件
    //     // int ov = read_input_graph(path + data_name + ".in", v_instance_graph);
    //     int ov = read_input_graph(path1, v_instance_graph);
    //     int V = v_instance_graph.size();  // 获取图的顶点总数
    //     int E = 0;                        // 边数计数器初始化
    //     std::cout << "read input complete" << endl;  // 输入图读取完成提示

    //     // 创建顶点到分组的映射表
    //     unordered_map<int,int>  map_v_to_group;

    //     // 读取分组文件并构建分组图结构
    //     read_Group(path + data_name + ".g", v_instance_graph, group_graph,map_v_to_group);
    //     std::cout << "read group complete " << group_graph.size() << endl;  // 分组读取完成提示

    //     srand(time(0));  // 用当前时间初始化随机种子
    //     // // 初始化随机数引擎（推荐使用 mt19937 算法）
    //     // std::random_device rd;    // 非确定性随机数种子[3,8](@ref)
    //     // std::mt19937 generator(rd());

    //     // // 定义均匀分布范围 [0.0, 60.0]
    //     // std::uniform_real_distribution<double> distribution(25,150);
    //     // 2. 定义闭区间范围 [min, max]
    //     // 循环生成sub_num个子图
    //     for(int s_num=0;s_num<sub_num;s_num++){
    //         vector<int> subgraph;  // 存储子图顶点集合
    //         E=0;
    //         // 声明子图结构和相关映射表
    //         graph_v_of_v_idealID v_sub_graph,v_sub_generated_group_graph;
    //         unordered_map<int,int> ordered_map_v_to_v,map_g_to_ordered_g,map_ordered_g_to_g,map_new_v_to_group;

    //         // 使用BFS随机寻找符合条件的子图
    //         while(1){
    //             int start_node = rand() % V;  // 随机选择起始节点
    //             subgraph.clear();  // 清空当前子图
    //             if(bfs_subgraph(start_node ,target_size,subgraph,v_instance_graph)) break;  // 找到足够大的子图时退出循环
    //         }
    //         std::cout<<"find subgraph "<<endl;  // 子图查找完成提示

    //         // 构建子图顶点到新编号的映射（0~999）
    //         int tmp=0;
    //         for(int v : subgraph){
    //             ordered_map_v_to_v[v]=tmp++;  // 原节点ID映射到连续编号
    //         }

    //         // 初始化子图邻接表和组扩展图结构
    //         v_sub_graph.resize(target_size);
    //         v_sub_generated_group_graph.resize(target_size);

    //         double weight_sum=0;
    //         // 构建子图的邻接关系
    //         for(int v : subgraph){
    //             int node = ordered_map_v_to_v[v];  // 获取新节点编号
    //             for(auto t : v_instance_graph[v]){  // 遍历原图邻接节点
    //                 if(ordered_map_v_to_v.find(t.first)==ordered_map_v_to_v.end()) continue;  // 跳过不在子图中的节点
    //                 int neighbor = ordered_map_v_to_v[t.first];  // 邻接节点的新编号
    //                 double w = t.second;
    //                 v_sub_graph[node].push_back({neighbor,w});  // 添加子图边
    //                 v_sub_generated_group_graph[node].push_back({neighbor,w});  // 添加组扩展图边
    //                 E++;  // 边数统计（注意后续会除以2）
    //                 weight_sum+=w;
    //             }
    //         }
    //         E/=2;  // 计算实际边数（无向图去重）
    //         weight_sum/=2;
    //         std::cout<<"E "<< E <<"weight_sum "<< weight_sum <<endl;
    //         weight_sum = weight_sum/E;
    //         std::cout<<"weight_sum "<< weight_sum <<endl;
    //         if(E>=15000){
    //             std::cout << "subgraph is too big" << endl;
    //             s_num--;
    //             continue;
    //         }
    //         // 对邻接表按节点编号排序
    //         for (size_t i = 0; i < v_sub_graph.size(); i++){
    //             std::sort(v_sub_graph[i].begin(), v_sub_graph[i].end());
    //         }

    //         // 构建组号到新编号的映射
    //         tmp=0;
    //         for(int v : subgraph){
    //             int group = map_v_to_group[v];  // 获取原组号
    //             if(map_g_to_ordered_g.find(group)==map_g_to_ordered_g.end()){
    //                 map_g_to_ordered_g[group] = tmp;    // 新组号分配
    //                 map_ordered_g_to_g[tmp] = group;   // 反向映射
    //                 tmp++;
    //             }
    //         }

    //         // 扩展组图结构（增加组节点）
    //         v_sub_generated_group_graph.resize(target_size+map_ordered_g_to_g.size());

    //         // 建立顶点与组节点的连接
    //         for(int v : subgraph){
    //             int node = ordered_map_v_to_v[v];
    //             int group = map_v_to_group[v];
    //             int ordered_group = map_g_to_ordered_g[group];  // 获取新组号
    //             map_new_v_to_group[node] = ordered_group;  // 记录顶点的新组号

    //             // 在组扩展图中添加顶点与组节点的双向边
    //             int ordered_group_vertex = target_size+ordered_group;  // 组节点编号（1000+组号）
    //             v_sub_generated_group_graph[node].push_back({ordered_group_vertex,1});
    //             v_sub_generated_group_graph[ordered_group_vertex].push_back({node,1});
    //         }
    //         cout<<"generate subgraph finish"<<endl;  // 子图生成完成提示

    //         // 创建T个查询任务
    //         cout<<"start generate task"<<endl;
    //         vector<unordered_set<int>> query_group_tasks(T);  // 存储每组查询任务

    //         for(int i=0;i<T;++i){
    //             cout<<"task "<<i<<" begin"<<endl;
    //             // 生成包含gamma个唯一组节点的查询任务
    //             while(query_group_tasks[i].size()<gamma){
    //                 // 随机选择组节点（编号范围：1000~1000+组数-1）
    //                 query_group_tasks[i].insert(target_size+rand()%map_ordered_g_to_g.size());
    //             }

    //             // 创建组号到标签的映射（1~gamma）
    //             unordered_map<int,int> g_to_label;
    //             int tmp=1;
    //             for(auto it : query_group_tasks[i]){
    //                 g_to_label[it] = tmp++;
    //             }

    //             // 调用PrunedDP++算法求解
    //             int RAM;  // 未初始化的内存参数（可能由算法填充）
    //             int p_cpu = 0;
    //             bool find=true;
    //             graph_hash_of_mixed_weighted solu = graph_v_of_v_idealID_PrunedDPPlusPlus(
    //                 v_sub_graph,
    //                 v_sub_generated_group_graph,
    //                 query_group_tasks[i],
    //                 1,
    //                 RAM,
    //                 &p_cpu,  // 未使用的CPU参数
    //                 find
    //             );

    //             if(!find) {  // 算法未找到解时重新生成任务
    //                 query_group_tasks[i].clear();
    //                 i--;
    //                 continue;
    //             }
    //             // double cost = cal_cost(v_sub_graph, solu);
    //             // if(cost >=300){
    //             //     i--;
    //             //     continue;
    //             // }
    //             // 写入数据文件
    //             data_file<<"Label ";
    //             // 写入每个顶点的组标签
    //             for(auto v:subgraph){
    //                 data_file<<g_to_label[target_size+map_g_to_ordered_g[map_v_to_group[v]]]<<" ";
    //             }
    //             data_file<<"|";
    //             // 写入组数量信息
    //             data_file<<"GroupNum "<<map_ordered_g_to_g.size()<<" ";
    //             // 写入查询组信息（原始组号）
    //             data_file<<"Query ";
    //             for(auto it : query_group_tasks[i]){
    //                 data_file<<it-1000<<" ";  // 转换为原始组号
    //             }
    //             // 写入顶点的新组号映射
    //             data_file<<"Group ";
    //             for(auto v:subgraph){
    //                 data_file<<map_g_to_ordered_g[map_v_to_group[v]]<<" ";
    //             }
    //             data_file<<"|";
    //             // 写入子图结构（邻接表）
    //             data_file <<"Graph ";
    //             int e1=0;
    //             for(int ii=0;ii<v_sub_graph.size();++ii){
    //                 for(auto t : v_sub_graph[ii]){
    //                     data_file << ii << " "<<t.first<<" "<<t.second<<" ";
    //                     e1++;
    //                 }
    //             }
    //             cout << e1 <<endl;
    //             // 写入算法求解结果
    //             data_file<<"Result ";
    //             for(auto it : solu.hash_of_vectors){
    //                 for(auto its : it.second.adj_vertices){
    //                     data_file<<it.first<<" "<<its.first<<" ";
    //                 }
    //             }
    //             data_file<<endl;
    //             cout<<"task "<<i<<" end"<<endl;
    //         }
    //     }
    //     // 关闭文件流
    //     data_file.close();
    //     cout<<"finish"<<endl;
    // }
    // cnt++;
    // Baseline
    // Baseline_file<<"GroupNum "<<map_ordered_g_to_g.size()<<" ";
    // Baseline_file<<"Query ";
    // for(auto it : query_group_tasks[i]){
    //     Baseline_file<<it-1000<<" ";
    // }

    // Baseline_file<<"Group ";
    // for(auto v:subgraph){
    //     Baseline_file<<map_g_to_ordered_g[map_v_to_group[v]]<<" ";
    // }
    // Baseline_file <<"Graph ";
    // for(int ii=0;ii<v_sub_graph.size();++ii){
    //     auto it = v_sub_graph[ii];
    //     for(auto t : it){
    //         Baseline_file << ii << " "<<t.first<<" "<<t.second<<" ";
    //     }
    // }
    // Baseline_file<<"Result ";
    // for(auto it : solu.hash_of_vectors){
    //     for(auto its : it.second.adj_vertices){
    //         Baseline_file<<it.first<<" "<<its.first<<" ";
    //     }
    // }
    // Baseline_file<<endl;
    // cout<<"task "<<i<<" end"<<endl;

    // //GNN

    // GNN_file<<"Label ";
    // for(auto v:subgraph){
    //     GNN_file<<g_to_label[target_size+map_g_to_ordered_g[map_v_to_group[v]]]<<" ";
    // }
    // GNN_file <<"Graph ";
    // for(int ii=0;ii<v_sub_graph.size();++ii){
    //     auto it = v_sub_graph[ii];
    //     for(auto t : it){
    //         GNN_file << ii << " "<<t.first<<" "<<t.second<<" ";
    //     }
    // }
    // GNN_file<<"Result ";
    // for(auto it : solu.hash_of_vectors){
    //     for(auto its : it.second.adj_vertices){
    //         GNN_file<<it.first<<" "<<its.first<<" ";
    //     }
    // }
    // GNN_file<<endl;
    // cout<<"task "<<i<<" end"<<endl;
    // }

    // }
    // GNN_file.close();
    // Baseline_file.close();
    // data_file.close();
    // cout<<"finish"<<endl;
    // }

    // // 生成一个随机整数
    // int G = 5, g_size_min = 2, g_size_max = 5, V = 50, E = 500, precision = 0;
    // double nw_min = 0, nw_max = 0, ec_min = 1, ec_max = 100;
    // double diff=0;
    // int cnt=0;
    // int iteration=100;
    // ofstream file("../GST-50/GST50-500.txt");
    // for(int k=0; k<5;k++){
    //     graph_hash_of_mixed_weighted instance_graph = graph_hash_of_mixed_weighted_generate_random_connected_graph(V,E,nw_min,nw_max,ec_min,ec_max,precision);
    //     for(int j=0;j<iteration;j++){
    //         G=dis(gen);

    //         bool find=true;
    //         bool find_noise=true;
    //         unordered_map<int,int> labels;
    //         auto [graph,graph_noise,ans,ans_noise] = prepare_data_depend_on_instance_graph(G, g_size_min, g_size_max, V, E, nw_min, nw_max, ec_min, ec_max, precision,labels,find,find_noise,instance_graph);
    //         if(!find||!find_noise||ans.hash_of_vectors.empty()) continue;
    //         // double cost=cal_cost(graph,ans);
    //         // double cost_noise=cal_cost(graph,ans_noise);
    //         // diff+=fabs((cost_noise-cost)/cost);
    //         // graph_hash_of_mixed_weighted graphs=res.first,ans=res.second;

    //         file<<"Label ";
    //         for(int i=0;i<labels.size();i++){
    //             file<<labels[i]<<" ";
    //         }
    //         file<<"Graph ";
    //         for(auto it : graph.hash_of_vectors){
    //             for(auto its : it.second.adj_vertices){
    //                 file<<it.first<<" "<<its.first<<" "<<its.second<<" ";
    //             }
    //         }
    //         // for(int i=0;i<graph.hash_of_vectors.size();i++){
    //         //     for(int j=0;j<graph.hash_of_vectors[i].adj_vertices.size();j++){
    //         //         file<<i<<" "<<graph.hash_of_vectors[i].adj_vertices[j].first<<" "<<graph.hash_of_vectors[i].adj_vertices[j].second<<" "<<graph_noise.hash_of_vectors[i].adj_vertices[j].second<<" ";
    //         //     }
    //         // }
    //         file<<"Result ";
    //         for(auto it : ans.hash_of_vectors){
    //             for(auto its : it.second.adj_vertices){
    //                 file<<it.first<<" "<<its.first<<" ";
    //             }
    //         }
    //         file<<endl;
    //         cnt++;
    //         if(j%100==0){
    //             cout<<"Complete : "<<j+k*iteration<<"/"<<5*iteration<<endl;
    //         }
    //     }
    //     // file<<"--------------------------split line--------------------------"<<endl;
    // }
    // file.close();
    // // cout<<"average diff : "<<diff/cnt<<endl;
    // return 0;
}
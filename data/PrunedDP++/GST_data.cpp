#include "rucgraph/GST_data.hpp"
#include <iostream>
#include <cmath>
#include <random>
using namespace std;

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

void generateData() 
{
    // Subgraph Generation
    cout << "start.." << endl;     
    int target_size = 1000;        

    int sub_num = 6;   
    int T = 20;        
    int gamma = 5;     
    std::cout << "Please enter the number of topological structures generating subgraphs: ";
    std::cin >> sub_num;
    
    std::cout << "Please enter the number of query tasks for each subgraph: ";
    std::cin >> T;
    
    std::cout << "Please enter the number of groups contained in each query task: ";
    std::cin >> gamma;
    for(int j=0;j<1;++j){
        string path_w="";  
        string path_g="";
        string data_path="";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "\nPlease enter the path to the source file of weight: ";
        std::getline(std::cin, path_w);
        std::cout << "Please enter the path to the source file of group: ";
        std::getline(std::cin, path_g);
        std::cout << "Please enter the path to the target file: ";
        std::getline(std::cin, data_path);
        if (path_w.empty()) {
            std::cerr << "Error: Source file of weight path not specified\n";
            return ;
        }
        if (path_g.empty()) {
            std::cerr << "Error: Source file of group path not specified\n";
            return ;
        }
        if (data_path.empty()) {
            std::cerr << "Error: Target file path not specified\n";
            return ;
        }
        ofstream data_file(data_path);  
        graph_v_of_v_idealID group_graph, v_instance_graph;
        int ov = read_input_graph(path_w, v_instance_graph);
        int V = v_instance_graph.size();  
        int E = 0;                        
        std::cout << "read input complete" << endl;  

        unordered_map<int,int>  map_v_to_group;

        read_Group(path_g, v_instance_graph, group_graph,map_v_to_group);
        std::cout << "read group complete " << group_graph.size() << endl;  

        srand(time(0));  
        for(int s_num=0;s_num<sub_num;s_num++){
            vector<int> subgraph;  
            E=0;
            graph_v_of_v_idealID v_sub_graph,v_sub_generated_group_graph;
            unordered_map<int,int> ordered_map_v_to_v,map_g_to_ordered_g,map_ordered_g_to_g,map_new_v_to_group;
            while(1){
                int start_node = rand() % V;  
                subgraph.clear();  
                if(bfs_subgraph(start_node ,target_size,subgraph,v_instance_graph)) break;  
            }
            std::cout<<"find subgraph "<<endl;  
            int tmp=0;
            for(int v : subgraph){
                ordered_map_v_to_v[v]=tmp++;  
            }
            v_sub_graph.resize(target_size);
            v_sub_generated_group_graph.resize(target_size);

            double weight_sum=0;
            for(int v : subgraph){
                int node = ordered_map_v_to_v[v];  
                for(auto t : v_instance_graph[v]){  
                    if(ordered_map_v_to_v.find(t.first)==ordered_map_v_to_v.end()) continue; 
                    int neighbor = ordered_map_v_to_v[t.first];  
                    double w = t.second;
                    v_sub_graph[node].push_back({neighbor,w}); 
                    v_sub_generated_group_graph[node].push_back({neighbor,w});  
                    E++;  
                    weight_sum+=w;
                }
            }
            E/=2;  
            weight_sum/=2;
            std::cout<<"E "<< E <<"weight_sum "<< weight_sum <<endl;
            weight_sum = weight_sum/E;
            std::cout<<"weight_sum "<< weight_sum <<endl;
            if(E>=15000){
                std::cout << "subgraph is too big" << endl;
                s_num--;
                continue;
            }
            for (size_t i = 0; i < v_sub_graph.size(); i++){
                std::sort(v_sub_graph[i].begin(), v_sub_graph[i].end());
            }
            tmp=0;
            for(int v : subgraph){
                int group = map_v_to_group[v];  
                if(map_g_to_ordered_g.find(group)==map_g_to_ordered_g.end()){
                    map_g_to_ordered_g[group] = tmp;    
                    map_ordered_g_to_g[tmp] = group;   
                    tmp++;
                }
            }
            v_sub_generated_group_graph.resize(target_size+map_ordered_g_to_g.size());
            for(int v : subgraph){
                int node = ordered_map_v_to_v[v];
                int group = map_v_to_group[v];
                int ordered_group = map_g_to_ordered_g[group];  
                map_new_v_to_group[node] = ordered_group; 
                int ordered_group_vertex = target_size+ordered_group; 
                v_sub_generated_group_graph[node].push_back({ordered_group_vertex,1});
                v_sub_generated_group_graph[ordered_group_vertex].push_back({node,1});
            }
            cout<<"generate subgraph finish"<<endl; 
            cout<<"start generate task"<<endl;
            vector<unordered_set<int>> query_group_tasks(T);  

            for(int i=0;i<T;++i){
                cout<<"task "<<i<<" begin"<<endl;
                while(query_group_tasks[i].size()<gamma){
                    query_group_tasks[i].insert(target_size+rand()%map_ordered_g_to_g.size());
                }
                unordered_map<int,int> g_to_label;
                int tmp=1;
                for(auto it : query_group_tasks[i]){
                    g_to_label[it] = tmp++;
                }
                int RAM;  
                int p_cpu = 0;
                bool find=true;
                graph_hash_of_mixed_weighted solu = graph_v_of_v_idealID_PrunedDPPlusPlus(
                    v_sub_graph,
                    v_sub_generated_group_graph,
                    query_group_tasks[i],
                    1,
                    RAM,
                    &p_cpu,  
                    find
                );

                if(!find) {  
                    query_group_tasks[i].clear();
                    i--;
                    continue;
                }
                
                data_file<<"Label ";
                for(auto v:subgraph){
                    data_file<<g_to_label[target_size+map_g_to_ordered_g[map_v_to_group[v]]]<<" ";
                }
                data_file<<"|";
                data_file<<"GroupNum "<<map_ordered_g_to_g.size()<<" ";
                data_file<<"Query ";
                for(auto it : query_group_tasks[i]){
                    data_file<<it-1000<<" ";  
                }
                data_file<<"Group ";
                for(auto v:subgraph){
                    data_file<<map_g_to_ordered_g[map_v_to_group[v]]<<" ";
                }
                data_file<<"|";
                data_file <<"Graph ";
                int e1=0;
                for(int ii=0;ii<v_sub_graph.size();++ii){
                    for(auto t : v_sub_graph[ii]){
                        data_file << ii << " "<<t.first<<" "<<t.second<<" ";
                        e1++;
                    }
                }
                cout << e1 <<endl;
                data_file<<"Result ";
                for(auto it : solu.hash_of_vectors){
                    for(auto its : it.second.adj_vertices){
                        data_file<<it.first<<" "<<its.first<<" ";
                    }
                }
                data_file<<endl;
                cout<<"task "<<i<<" end"<<endl;
            }
        }
        data_file.close();
        cout<<"finish"<<endl;
    }
}

void executeAlgorithm()
{
    cout << "start.." << endl;
    int target_size = 1000;
    double total_error = 0;
    int Num = 0;
    int gamma = 3;
    string dataName = "";
    string path = "";
    std::cout << "Please enter the number of groups contained in each query task: ";
    std::cin >> gamma;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cout << "\nPlease enter a test file: ";
    std::getline(std::cin, dataName);
    std::cout << "\nPlease enter the result storage directory: ";
    std::getline(std::cin, path);
    if (path.empty() || dataName.empty()) {
        std::cerr << "Error: file name and path must be provided! " << std::endl;
        return ;
    }
    std::filesystem::path dir_path(path);
    std::filesystem::create_directories(dir_path);
    ofstream result(dir_path / "PrunedDP_result.txt");
    result << dataName+" result : " << endl;

    for (int gammaNum = 1; gammaNum < 2; gammaNum++)
    {
        if (gamma == 3)
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
            double total_duration = 0;     
            double total_length = 0;      
            double total_length_noise = 0;
            string filename = dataName;

            std::ifstream myfile(filename);
            string linecontent;
            int count_line = 0;
            if (myfile.is_open()) 
            {
                int line = 0;
                string tmp;
                double sum_cost = 0;
                double error = 0;
                int count = 0;
                int E = 0;

                while (getline(myfile, linecontent)) 
                {
                    auto start = std::chrono::high_resolution_clock::now();
                    line++;
                    graph_v_of_v_idealID v_sub_graph, v_sub_generated_group_graph, v_sub_generated_group_graph_noise, v_sub_graph_noise;
                    vector<pair<int, int>> ans;
                    unordered_set<int> query_group_vertex;
                    vector<int> v_to_g;
                    size_t split_pos = linecontent.find('|');
                    string result1 = linecontent.substr(split_pos + 1);
                    std::istringstream inputStream(result1);

                    string t;
                    int GroupNum;

                    while (inputStream >> tmp)
                    {
                        if (!tmp.compare("GroupNum"))
                        {
                            inputStream >> t;
                            GroupNum = std::stoi(t);
                            v_sub_graph.resize(target_size);
                            v_sub_graph_noise.resize(target_size);
                            v_sub_generated_group_graph.resize(target_size + GroupNum);
                            v_sub_generated_group_graph_noise.resize(target_size + GroupNum);
                        }
                        else if (!tmp.compare("Query"))
                        {
                            for (int i = 0; i < Num; ++i)
                            {
                                inputStream >> t;
                                query_group_vertex.insert(target_size + std::stoi(t));
                            }
                        }
                        else if (!tmp.compare("Group"))
                        {
                            for (int i = 0; i < target_size; ++i)
                            {
                                inputStream >> t;
                                int g = std::stoi(t);
                                v_to_g.push_back(g);
                                v_sub_generated_group_graph[i].push_back({target_size + g, 1});
                                v_sub_generated_group_graph[target_size + g].push_back({i, 1});
                                v_sub_generated_group_graph_noise[i].push_back({target_size + g, 1});
                                v_sub_generated_group_graph_noise[target_size + g].push_back({i, 1});
                            }
                        }
                        else if (!tmp.compare("|Graph"))
                        {
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
                            while (inputStream >> t)
                            {
                                int u = std::stoi(t);
                                inputStream >> t;
                                int v = std::stoi(t);
                                ans.push_back({u, v});
                            }
                        }
                    }
                    double cost = 0;
                    double cost_noise = 0;
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
                    count_line++;
                    cost_noise = cal_cost(v_sub_graph, solu);
                    result << cost_noise << "   " << cost << endl;
                    double d = fabs((cost - cost_noise) / cost);
                    
                    total_length_noise += cost_noise;
                    sum_cost += cost_noise;
                    auto end = std::chrono::high_resolution_clock::now();

                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                    cout << filename << "   line: " << line << " error = " << d * 100 << " \% " << "cost " << duration.count() / 1000 << " ms" << endl;
                    total_duration += duration.count() / 1000;
                }
                total_error = fabs((total_length / 600 - total_length_noise / count_line) / (total_length / 600));
                cout << filename << " finish " << count_line << "tasks   average error = " << total_error * 100 << " % with time " << total_duration << " ms" << endl;
            }
            result << filename << count_line << " tasks  error = " << total_error * 100 << " \% time = " << total_duration << "ms average_time = " << total_duration / count_line << "ms  length_actural = " << total_length / 600 << "   length_noise = " << total_length_noise / count_line << endl;
        }
    }
    result.close();
}

int main(int argc, char *argv[])
{
    int choice = -1;
    std::cout << "Please select the operation mode:" << std::endl;
    std::cout << "0 - Generate data" << std::endl;
    std::cout << "1 - Execute PrunedDP++ on the dataset" << std::endl;
    std::cout << "Please enter the option (0 or 1).";
    
    if (!(std::cin >> choice)) {
        std::cerr << "\n Error: Please enter a valid numeric option!" << std::endl;
        return 1;
    }

    switch(choice) {
        case 0:
            std::cout << "\nYou have selected the data generation mode.\n" << std::endl;
            generateData();
            break;
        case 1:
            std::cout << "\nYou have selected the algorithm execution mode.\n" << std::endl;
            executeAlgorithm();
            break;
        default:
            std::cerr << "\nError: Invalid option! Please enter 0 or 1." << std::endl;
            return 1;
    }

    return 0;
}

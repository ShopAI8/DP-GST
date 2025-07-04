#pragma once

#include <chrono>
#include <queue>
#include <omp.h>
#include <boost/heap/fibonacci_heap.hpp>
#include <graph_hash_of_mixed_weighted/graph_hash_of_mixed_weighted.h>
#include <graph_hash_of_mixed_weighted/common_algorithms/graph_hash_of_mixed_weighted_connected_components.h>
#include <graph_hash_of_mixed_weighted/random_graph/graph_hash_of_mixed_weighted_generate_random_connected_graph.h>
#include <graph_hash_of_mixed_weighted/two_graphs_operations/graph_hash_of_mixed_weighted_to_graph_v_of_v_idealID.h>
#include <graph_hash_of_mixed_weighted_generate_random_groups_of_vertices.h>
#include <graph_hash_of_mixed_weighted_save_for_GSTP.h>
#include <graph_hash_of_mixed_weighted_read_for_GSTP.h>
#include "graph_hash_of_mixed_weighted_sum_of_nw_ec.h"
#include "graph_v_of_v_idealID_DPBF_only_ec.h"
#include "graph_v_of_v_idealID_PrunedDPPlusPlus.h"
#include "DPQ.cuh"

non_overlapped_group_sets graph_v_of_v_idealID_DPBF_non_overlapped_group_sets_gpu(int group_sets_ID_range)
{
	non_overlapped_group_sets s;
	s.length = 0;
	s.non_overlapped_group_sets_IDs_pointer_host.resize(group_sets_ID_range + 3);
	/*this function calculate the non-empty and non_overlapped_group_sets_IDs of each non-empty group_set ID;
	time complexity: O(4^|Gamma|), since group_sets_ID_range=2^|Gamma|;
	the original DPBF code use the same method in this function, and thus has the same O(4^|Gamma|) complexity;*/
	// <set_ID, non_overlapped_group_sets_IDs>
	for (int i = 1; i <= group_sets_ID_range; i++)
	{ // i is a nonempty group_set ID
		s.non_overlapped_group_sets_IDs_pointer_host[i] = s.length;
		for (int j = 1; j < group_sets_ID_range; j++)
		{ // j is another nonempty group_set ID
			if ((i & j) == 0)
			{ // i and j are non-overlapping group sets
				/* The & (bitwise AND) in C or C++ takes two numbers as operands and does AND on every bit of two numbers. The result of AND for each bit is 1 only if both bits are 1.
				https://www.programiz.com/cpp-programming/bitwise-operators */
				s.non_overlapped_group_sets_IDs.push_back(j);
				s.length++;
			}
		}
	}
	s.non_overlapped_group_sets_IDs_pointer_host[group_sets_ID_range + 1] = s.length;
	std::cout << "len= " << s.length << std::endl;
	return s;
}
bool this_is_a_feasible_solution_gpu(graph_hash_of_mixed_weighted &solu, graph_hash_of_mixed_weighted &group_graph,
									 std::unordered_set<int> &group_vertices)
{

	/*time complexity O(|V_solu|+|E_solu|)*/
	if (graph_hash_of_mixed_weighted_connected_components(solu).size() != 1)
	{ // it's not connected
		cout << "this_is_a_feasible_solution: solu is disconnected!" << endl;
		return false;
	}

	for (auto it = group_vertices.begin(); it != group_vertices.end(); it++)
	{
		int g = *it;
		bool covered = false;
		for (auto it2 = solu.hash_of_vectors.begin(); it2 != solu.hash_of_vectors.end(); it2++)
		{
			int v = it2->first;
			if (graph_hash_of_mixed_weighted_contain_edge(group_graph, v, g))
			{
				covered = true;
				break;
			}
		}
		if (covered == false)
		{
			cout << "this_is_a_feasible_solution: a group is not covered!" << endl;
			return false;
		}
	}

	return true;
}

void test_graph_v_of_v_idealID_DPBF_only_ec_gpu()
{

	/*parameters*/
	int iteration_times = 10;
	int V = 300000, E = 800000, G = 6, g_size_min = 1, g_size_max = 4, precision = 0;
	int ec_min = 1, ec_max = 4; // PrunedDP does not allow zero edge weight

	int solution_cost_DPBF_sum = 0, solution_cost_PrunedDPPlusPlus_sum = 0;

	double time_DPBF_avg = 0, time_PrunedDPPlusPlus_avg = 0;
	int p_gpu = 0, p_cpu = 0;
	int *pointer1 = &p_gpu, *pointer2 = &p_cpu;
	int generate_new_graph = 0;
	int lambda = 1;
	std::unordered_set<int> generated_group_vertices;
	graph_hash_of_mixed_weighted instance_graph, generated_group_graph;
	if (generate_new_graph == 1)
	{
		instance_graph = graph_hash_of_mixed_weighted_generate_random_connected_graph(V, E, 0, 0, ec_min, ec_max, precision);

		graph_hash_of_mixed_weighted_generate_random_groups_of_vertices(G, g_size_min, g_size_max,
																		instance_graph, instance_graph.hash_of_vectors.size(), generated_group_vertices, generated_group_graph); //

		graph_hash_of_mixed_weighted_save_for_GSTP("simple_iterative_tests.text", instance_graph,
												   generated_group_graph, generated_group_vertices, lambda);
	}
	else
	{

		graph_hash_of_mixed_weighted_read_for_GSTP("movielens_data.txt", instance_graph,
												   generated_group_graph, generated_group_vertices, lambda);
	}
	cout << "generate complete" << endl;
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
	CSR_graph csr_graph = toCSR(v_instance_graph);
	cout << "E:" << csr_graph.E_all << " v:" << csr_graph.V << endl;
	int *community = new int[csr_graph.V], c_size[3];
	graph_hash_of_mixed_weighted_read_community("community", community, c_size);
	std::cout << "get community complete " << std::endl;
	CSR_graph part_graph[3];
	toCSR_three(part_graph, v_instance_graph, community, c_size);
	G = generated_group_vertices.size();
	int group_sets_ID_range = pow(2, G) - 1;
	non_overlapped_group_sets s = graph_v_of_v_idealID_DPBF_non_overlapped_group_sets_gpu(group_sets_ID_range);
	/*iteration*/
	for (int i = 0; i < iteration_times; i++)
	{

		cout << "iteration " << i << endl;

		/*input and output*/

		/*graph_v_of_v_idealID_DPBF_only_ec*/
		generated_group_graph.clear();
		graph_hash_of_mixed_weighted_generate_community_groups_of_vertices(G, g_size_min, g_size_max,
																		   instance_graph, instance_graph.hash_of_vectors.size(), generated_group_vertices, generated_group_graph, 50, community); //
		graph_v_of_v_idealID v_generated_group_graph = graph_hash_of_mixed_weighted_to_graph_v_of_v_idealID(generated_group_graph, vertexID_old_to_new);
		std::cout << "get inquire complete" << std::endl;
		if (1)
		{

			node **host_tree;
			int height = csr_graph.V, width = group_sets_ID_range + 1;
			host_tree = new node *[height];
			node *host_tree_one_d = new node[height * width];
			for (size_t i = 0; i < height; i++)
			{
				host_tree[i] = &host_tree_one_d[i * width];
			}
			int RAM, *real_cost;
			auto begin = std::chrono::high_resolution_clock::now();

			//  DPBF_GPU_part(part_graph[0],generated_group_vertices, v_generated_group_graph, v_instance_graph, pointer1, real_cost,community,c_size,1,s);
			//   DPBF_GPU_part(part_graph[1],generated_group_vertices, v_generated_group_graph, v_instance_graph, pointer1, real_cost,community,c_size,2,s);
			//   DPBF_GPU_part(part_graph[2],generated_group_vertices, v_generated_group_graph, v_instance_graph, pointer1, real_cost,community,c_size,3,s);
#pragma omp parallel for
			for (int i = 0; i < 3; ++i)
			{
				//DPBF_GPU_part(host_tree, part_graph[i], generated_group_vertices, v_generated_group_graph, v_instance_graph, pointer1, real_cost, community, c_size, i + 1, s);
			}

			graph_hash_of_mixed_weighted solu = DPBF_GPU(host_tree, host_tree_one_d, csr_graph, part_graph, generated_group_vertices, v_generated_group_graph, v_instance_graph, pointer1, real_cost, community, c_size, s);
			auto end = std::chrono::high_resolution_clock::now();
			double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
			time_DPBF_avg += (double)runningtime / iteration_times;

			// graph_hash_of_mixed_weighted_print(solu);
			int cost;
			real_cost = &cost;
			solution_cost_DPBF_sum += *real_cost;
			cost = graph_hash_of_mixed_weighted_sum_of_ec(solu);

			cout << "form tree cost: " << cost << endl;

			if (!this_is_a_feasible_solution_gpu(solu, generated_group_graph, generated_group_vertices))
			{
				cout << "Error: graph_v_of_v_idealID_DPBF_only_ec is not feasible!" << endl;
				graph_hash_of_mixed_weighted_print(solu);
				exit(1);
			}
		}

		/*graph_hash_of_mixed_weighted_PrunedDPPlusPlus_edge_weighted*/
		if (1)
		{
			int RAM;
			auto begin = std::chrono::high_resolution_clock::now();
			graph_hash_of_mixed_weighted solu = graph_v_of_v_idealID_PrunedDPPlusPlus(v_instance_graph, v_generated_group_graph, generated_group_vertices, 1, RAM, pointer2);
			auto end = std::chrono::high_resolution_clock::now();
			double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
			time_PrunedDPPlusPlus_avg = time_PrunedDPPlusPlus_avg + (double)runningtime / iteration_times;

			// graph_hash_of_mixed_weighted_print(solu);

			int cost = graph_hash_of_mixed_weighted_sum_of_ec(solu);
			solution_cost_PrunedDPPlusPlus_sum = solution_cost_PrunedDPPlusPlus_sum + cost;

			if (!this_is_a_feasible_solution_gpu(solu, generated_group_graph, generated_group_vertices))
			{
				cout << "Error: graph_v_of_v_idealID_DPBF_only_ec is not feasible!" << endl;
				graph_hash_of_mixed_weighted_print(solu);
				// exit(1);
			}
		}

		if (solution_cost_DPBF_sum + 1e-8 < solution_cost_PrunedDPPlusPlus_sum)
		{
			cout << "solution_cost_DPQ_GPU_sum=" << solution_cost_DPBF_sum << endl;
			cout << "solution_cost_PrunedDPPlusPlus_sum=" << solution_cost_PrunedDPPlusPlus_sum << endl;
			cout << "wrong answer" << endl;
			
		}
	}
	cout << "gpu " << *pointer1 << "cpu " << *pointer2 << "  " << (*pointer1) / (*pointer2) << endl;
	cout << "solution_cost_DPBF_sum=" << solution_cost_DPBF_sum << endl;
	cout << "solution_cost_PrunedDPPlusPlus_sum=" << solution_cost_PrunedDPPlusPlus_sum << endl;

	cout << "time_DPBF_avg=" << time_DPBF_avg << "s" << endl;
	cout << "time_PrunedDPPlusPlus_avg=" << time_PrunedDPPlusPlus_avg << "s" << endl;
	cout << time_PrunedDPPlusPlus_avg / time_DPBF_avg << endl;
}
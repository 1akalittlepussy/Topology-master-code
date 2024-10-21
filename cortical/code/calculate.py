import math
from platform import node
from time import time
from utils import *
import networkx as nx
import numpy as np
from networkx.algorithms import community
from result_struct import *

def calculate_dynamic_topology_by_source_mat(source_mat, thresh_persent=0.2, sliding_window_size=50):
    '''
        输入: 
            - source_mat, example: source_mat.shape = (200 * 90), 90个脑区, 200个时间序列
            - thresh_persent, 二值化百分比
            - sliding_window_size, 滑动窗口大小
        
        输出: 
            - result_dict = class.dynamic_result_struct
    '''

    # dynamic_result_struct结构
    result_dict = dynamic_result_struct()

    # 滑动窗口
    for start_index in range(0, np.shape(source_mat)[0]-sliding_window_size, 2):
        sub_source_mat = source_mat[start_index:start_index+sliding_window_size, :]

        single_result_dict = calculate_static_topology_by_source_mat(sub_source_mat)

        result_dict.adjacency_mats.append(single_result_dict['adjacency_mat'])

        # 添加每个窗口的结构信息到sub_result_sequence（图结构信息）
        for key_name in result_dict.info_in_graph.keys():
            if key_name in single_result_dict.keys():
                result_dict.info_in_graph[key_name]['sub_result_sequence'].append(np.array(single_result_dict[key_name]))

        # 添加每个窗口的结构信息到sub_result_sequence（节点结构信息）
        for key_name in result_dict.info_in_nodes.keys():
            if key_name in single_result_dict.keys():
                result_dict.info_in_nodes[key_name]['sub_result_sequence'].append(np.array(single_result_dict[key_name]))

    # 计算每个指标的均值、方差（图结构信息）
    for key_name in result_dict.info_in_graph.keys():
        result_dict.info_in_graph[key_name]['mean'] = np.mean(result_dict.info_in_graph[key_name]['sub_result_sequence'])
        result_dict.info_in_graph[key_name]['std'] = np.std(result_dict.info_in_graph[key_name]['sub_result_sequence'])

    # 计算每个指标的均值、方差（节点结构信息）
    for key_name in result_dict.info_in_nodes.keys():
        result_dict.info_in_nodes[key_name]['mean'] = np.mean(result_dict.info_in_nodes[key_name]['sub_result_sequence'], axis=0)
        result_dict.info_in_nodes[key_name]['std'] = np.std(result_dict.info_in_nodes[key_name]['sub_result_sequence'], axis=0)

    result_dict.source_mat = source_mat
    
    return result_dict


def calculate_static_topology_by_source_mat(source_mat, thresh_persent=0.2):
    '''
        输入: source_mat, example: source_mat.shape = (200 * 90), 90个脑区, 200个时间序列
        输出: dict(){degree, adjacency_mat, local_efficiency, global_efficiency, nodal_efficiency, clustering, avg_path_length, modularity_q}
    '''

    # 皮尔森相关系数
    related_mat = np.corrcoef(source_mat.T)

    # 排序，计算20%的阈值
    thresh_val = get_thresh_val(related_mat)

    # 二值化处理
    adjacency_mat = convert_binary_by_thresh_val(related_mat, thresh_val)

    return calculate_static_topology_by_adjacency_mat(adjacency_mat, thresh_persent)


def calculate_static_topology_by_adjacency_mat(adjacency_mat, thresh_persent=0.2):
    '''
        输入adjacency_mat
    '''
    
    # 构图
    G = nx.from_numpy_array(adjacency_mat)

    degree = cal_degree(G)

    local_efficiency, global_efficiency = cal_efficiency(G)

    nodal_efficiency = cal_nodal_efficiency(G)

    clustering, average_clustering = cal_clustering(G)

    avg_path_length = cal_harmonic_mean_L(G)

    modularity_q = cal_modularity_Q(G)

    # random_efficiency = cal_random_efficiency(G)

    # # small_world=0代表非小世界网络，small_world=1代表属于小世界网络
    # small_world = 0
    # if efficiency[0] / random_efficiency[0] > 1 and math.isclose(efficiency[1], random_efficiency[1], 1e-3):
    #     small_world = 1

    result_dict = dict()

    result_dict['degree'] = degree
    result_dict['adjacency_mat'] = adjacency_mat
    # result_dict['small_world'] = small_world
    result_dict['local_efficiency'] = local_efficiency
    result_dict['global_efficiency'] = global_efficiency
    result_dict['nodal_efficiency'] = nodal_efficiency
    result_dict['clustering'] = clustering
    result_dict['average_clustering'] = average_clustering
    result_dict['avg_path_length'] = avg_path_length
    result_dict['modularity_q'] = modularity_q
    # result_dict['random_efficiency'] = random_efficiency

    return result_dict


def cal_degree(graph):
    degree = nx.degree(graph)

    res = []

    for node_degree in degree:
        res.append(node_degree[1])
    
    return res

def cal_clustering(graph):
    clustering_for_nodes = nx.clustering(graph)
    nodes = list(nx.nodes(graph))

    nodes = np.sort(nodes, kind='mergesort')
    res = []

    for node_index in nodes:
        res.append(clustering_for_nodes[node_index])

    average_clustering = nx.average_clustering(graph)
    
    return res, average_clustering

def cal_small_world(test_graph):
    '''
        暂时该传统方法，采用efficiency计算方法
    '''
    return None
    # # 随即等效图
    # random_graph = nx.random_reference(test_graph)
    # # 等效网格
    # lattice_graph = nx.lattice_reference(test_graph)

    coefficient_sigma = nx.sigma(test_graph)

    coefficient_omega = nx.omega(test_graph)

    return (coefficient_sigma, coefficient_omega)


def cal_efficiency(test_graph):

    local_efficiency = nx.local_efficiency(test_graph)

    global_efficiency = nx.global_efficiency(test_graph)

    return local_efficiency, global_efficiency


def cal_random_efficiency(test_graph):
    randMetrics = {"global": [], "local": []}
    latticeMetrics = {"global": [], "local": []}
    for i in range(100):
        # 等效随机图
        Gr = nx.random_reference(test_graph)
        randMetrics["global"].append(nx.global_efficiency(Gr))
        randMetrics["local"].append(nx.local_efficiency(Gr))
        # print(randMetrics["local"][-1], randMetrics["global"][-1])

        # 等效网格
        # Gl = nx.lattice_reference(G)
        # latticeMetrics["global"].append(nx.global_efficiency(Gl))
        # latticeMetrics["local"].append(nx.local_efficiency(Gl))

        # print(latticeMetrics["local"][-1], latticeMetrics["global"][-1])

    # 平均global_efficiency和local_efficiency
    G_rand_global = np.mean(randMetrics['global'])
    G_rand_local = np.mean(randMetrics['local'])

    # G_lattice_global = np.mean(latticeMetrics['global'])
    # G_lattice_local = np.mean(latticeMetrics['local'])

    return (G_rand_local, G_rand_global)

def cal_nodal_efficiency(graph):
    nodes = list(nx.nodes(graph))
    N = len(nodes)

    nodal_efficiency = [[0] * N] * N 

    for i in range(N):
        for j in range(i+1, N):
            node_i = nodes[i]
            node_j = nodes[j]

            nodal_efficiency_i_j = nx.efficiency(graph, node_i, node_j) 

            nodal_efficiency[i][j] = nodal_efficiency_i_j
            nodal_efficiency[j][i] = nodal_efficiency_i_j

    nodal_efficiency_res = np.sum(nodal_efficiency, axis=0) / (N -1)
    
    return nodal_efficiency_res

def cal_harmonic_mean_L(graph):
    harmonic_centrality_nodes_dict = nx.harmonic_centrality(graph)

    nodes = list(nx.nodes(graph))
    N = len(nodes)

    total_path_length = 0
    for node in nodes:
        total_path_length += harmonic_centrality_nodes_dict[node]

    return (N * (N - 1)) / (total_path_length / 2)

def cal_modularity_Q(graph):
    communities = community.greedy_modularity_communities(graph)

    Q = community.modularity(graph, communities)

    return Q
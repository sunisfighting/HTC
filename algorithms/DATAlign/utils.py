from evaluation.metrics import get_statistics
import numpy as np
import torch
import pickle
from scipy.sparse import coo_matrix
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_equivalent_edges(source_edges, target_edges, full_dict):
    count_edges = 0
    source_edges_list = []
    target_edges_list = []
    source_edges = source_edges.tolist()
    target_edges = target_edges.tolist()
    while count_edges < 100:
        index = np.random.randint(0, len(source_edges), 1)[0]
        source_edge = source_edges[index]
        if source_edge not in source_edges_list:
            first_node = source_edge[0]
            second_node = source_edge[1]
            try:
                first_node_target = full_dict[first_node]
                second_node_target = full_dict[second_node]
            except:
                continue
            if [first_node_target, second_node_target] in target_edges:
                source_edges_list.append(source_edge)
                target_edges_list.append([first_node_target, second_node_target])
                count_edges += 1
    
    source_nodes = np.random.choice(np.array(list(full_dict.keys())), 100, replace=False)
    target_nodes = np.array([full_dict[source_nodes[i]] for i in range(len(source_nodes))])

    return source_edges_list, target_edges_list, source_nodes, target_nodes

def investigate(source_outputs, target_outputs, source_edges, target_edges, full_dict):
    source_edges, target_edges, source_nodes, target_nodes = get_equivalent_edges(source_edges, target_edges, full_dict)
    source_edges_np = np.array(source_edges)
    target_edges_np = np.array(target_edges)

    source_nodes_np = np.array(source_nodes)
    target_nodes_np = np.array(target_nodes)
    first_source_nodes_np = source_edges_np[:, 0]
    second_source_nodes_np = source_edges_np[:, 1]
    first_target_nodes_np = target_edges_np[:, 0]
    second_target_nodes_np = target_edges_np[:, 1]

    source_nodes_tensor = torch.LongTensor(source_nodes_np).cuda()
    target_nodes_tensor = torch.LongTensor(target_nodes_np).cuda()
    first_source_nodes_tensor = torch.LongTensor(first_source_nodes_np).cuda()
    second_source_nodes_tensor = torch.LongTensor(second_source_nodes_np).cuda()
    first_target_nodes_tensor = torch.LongTensor(first_target_nodes_np).cuda()
    second_target_nodes_tensor = torch.LongTensor(second_target_nodes_np).cuda() 

    source_nodes_emb = [source_outputs[i][source_nodes_tensor] for i in range(len(source_outputs))]
    target_nodes_emb = [target_outputs[i][target_nodes_tensor] for i in range(len(source_outputs))]
    first_source_nodes_emb = [source_outputs[i][first_source_nodes_tensor] for i in range(len(source_outputs))]
    second_source_nodes_emb = [source_outputs[i][second_source_nodes_tensor] for i in range(len(source_outputs))]
    first_target_nodes_emb = [target_outputs[i][first_target_nodes_tensor] for i in range(len(source_outputs))]
    second_target_nodes_emb = [target_outputs[i][second_target_nodes_tensor] for i in range(len(source_outputs))]

    edges_distance_source = [torch.sum((first_source_nodes_emb[i] - second_source_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]
    edges_distance_target = [torch.sum((first_target_nodes_emb[i] - second_target_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]
    anchor_distance1 = [torch.sum((first_source_nodes_emb[i] - first_target_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]
    anchor_distance2 = [torch.sum((second_source_nodes_emb[i] - second_target_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]
    random_distance1 = [torch.sum((first_source_nodes_emb[i] - source_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]
    random_distance2 = [torch.sum((first_target_nodes_emb[i] - target_nodes_emb[i])**2, dim=1) for i in range(len(source_outputs))]

    for i in range(len(edges_distance_source)):
        print("Layer: {}, edge source: {:.4f}, edge target: {:.4f}, non edge source: {:.4f}, non edge target: {:.4f}".format(i, edges_distance_source[i].mean(), edges_distance_target[i].mean(), \
                random_distance1[i].mean(), random_distance2[i].mean()))
        print("Layer: {}, anchor distance1: {:.4f}, anchor distance2: {:.4f}".format(i, anchor_distance1[i].mean(), anchor_distance2[i].mean()))


def get_acc(source_outputs, target_outputs, test_dict = None, alphas=None, just_S = False):
    Sf = np.zeros((len(source_outputs[0]), len(target_outputs[0])))  #第一个隐层的特征维度
    list_S_numpy = []
    accs = ""
    for i in range(0, len(source_outputs)):  #逐层
        S = torch.matmul(F.normalize(source_outputs[i]), F.normalize(target_outputs[i]).t())
        S_numpy = S.detach().cpu().numpy()
        if test_dict is not None:
            if not just_S:
                acc = get_statistics(S_numpy, test_dict)
                accs += "Acc layer {} is: {:.4f}, ".format(i, acc)
        if alphas is not None:
            Sf += alphas[i] * S_numpy
        else:
            Sf += S_numpy
    if test_dict is not None:
        if not just_S:
            acc = get_statistics(Sf, test_dict)
            accs += "Final acc is: {:.4f}".format(acc)
    return accs, Sf


def fixed_unigram_candidate_sampler(num_sampled, unique, range_max, distortion, unigrams):
    weights = unigrams**distortion
    prob = weights/weights.sum()
    sampled = np.random.choice(range_max, num_sampled, p=prob, replace=~unique)
    return sampled

def Laplacian_graph(A):
    for i in range(len(A)):
        A[i, i] = 1
    A = torch.FloatTensor(A)
    D_ = torch.diag(torch.sum(A, 0)**(-0.5))
    A_hat = torch.matmul(torch.matmul(D_,A),D_)
    A_hat = A_hat.float()
    indices = torch.nonzero(A_hat).t()
    values = A_hat[indices[0], indices[1]]
    A_hat = torch.sparse.FloatTensor(indices, values, A_hat.size())
    return A_hat, coo_matrix(A.detach().cpu().numpy())

def update_Laplacian_graph(old_A, new_edges):
    count_updated = 0
    for edge in new_edges:
        if old_A[edge[0], edge[1]] == 0:
            count_updated += 1
        old_A[edge[0], edge[1]] = 1
        old_A[edge[1], edge[0]] = 1
    new_A_hat, new_A = Laplacian_graph(old_A)
    print("Updated {} edges".format(count_updated))
    return new_A_hat, new_A


def get_candidate_edges(S, edges, threshold):
    S = S / 3
    points_source, points_source_index = S[edges[:, 0]].max(dim=1)
    points_target, points_target_index = S[edges[:, 1]].max(dim=1)
    new_edges = []
    for i in range(len(points_source)):
        point_source = points_source[i]
        point_target = points_target[i]
        if point_source > threshold and point_target > threshold:
            new_edges.append((points_source_index[i], points_target_index[i]))
    return new_edges


def get_source_target_neg(args, source_deg, target_deg, source_edges, target_edges):
    source_negs = []
    target_negs = []
    for i in range(0, len(source_edges), 512):
        source_neg = fixed_unigram_candidate_sampler(
                num_sampled=args.neg_sample_size,
                unique=False,
                range_max=len(source_deg),
                distortion=0.75,
                unigrams=source_deg
                )
        
        source_neg = torch.LongTensor(source_neg).cuda()
        source_negs.append(source_neg)

    for i in range(0 ,len(target_edges), 512):

        target_neg = fixed_unigram_candidate_sampler(
            num_sampled=args.neg_sample_size,
            unique=False,
            range_max=len(target_deg),
            distortion=0.75,
            unigrams=target_deg
            )

        target_neg = torch.LongTensor(target_neg).cuda()
        target_negs.append(target_neg)

    return source_negs, target_negs       


def save_embeddings(source_outputs, target_outputs):
    print("Saving embeddings")
    for i in range(len(source_outputs)):
        ele_source = source_outputs[i]
        ele_source = ele_source.detach().cpu().numpy()
        ele_target = target_outputs[i]
        ele_target = ele_target.detach().cpu().numpy()
        np.save("numpy_emb/source_layer{}".format(i), ele_source)
        np.save("numpy_emb/target_layer{}".format(i), ele_target)
    print("Done saving embeddings")


def investigate_similarity_matrix(S, full_dict, source_deg, target_deg, source_edges, target_edges):
    """
    Source info:
    - Nodes:  3906 (3.5 times target nodes)
    - Edges:  8164 (5.4 times target edges)
    - Edges/node: 2 (1.42 times)
    Target info: (Smaller than Source but edges are closer than source)
    - Nodes:  1118
    - Edges:  1511
    - Edges/node: 1.4

    after train:
    Layer: 0, edge source: 1.0600, edge target: 1.0600, non edge source: 1.6800, non edge target: 1.6800
    Layer: 1, edge source: 0.8366 (1.38 times), edge target: 0.6058, non edge source: 1.8595, non edge target: 1.8326
    Layer: 2, edge source: 0.6010 (1.51 times), edge target: 0.3970, non edge source: 1.7425, non edge target: 1.7834
    Layer: 3, edge source: 0.4916 (1.82 times), edge target: 0.2689, non edge source: 1.7873, non edge target: 1.7470


    Layer: 0, anchor distance1: 0.0000, anchor distance2: 0.0000
    Layer: 1, anchor distance1: 0.3191, anchor distance2: 0.3638
    Layer: 2, anchor distance1: 0.2811, anchor distance2: 0.3047
    Layer: 3, anchor distance1: 0.3040, anchor distance2: 0.3799

    what do I want to know???
    At each layer, which ones are match??? (Save match at each layer as list)
    Visualize source anchor nodes (save)
    Visualize target anchor nodes (save)
    Visualize match node at each layer

    Degree distribution of matched nodes at each layer


    """
    source_nodes = np.array(list(full_dict.keys()))
    target_nodes = np.array(list(full_dict.values()))
    hits_source = []
    hits_target = []
    for i in range(len(S)):
        S_i = S[i][source_nodes]
        matched_source = np.argmax(S_i, axis=1)
        hit_i_source = []
        hit_i_target = []
        for j in range(len(source_nodes)):
            if matched_source[j] == target_nodes[j]:
                hit_i_source.append(source_nodes[j])
                hit_i_target.append(target_nodes[j])
        hits_source.append(hit_i_source)
        hits_target.append(hit_i_target)
    
    tosave = [hits_source, hits_target]
    with open("douban_data", "wb") as f:
        pickle.dump(tosave, f)

    
    for i in range(len(hits_source)):
        source_deg_i = np.array([source_deg[k] for k in hits_source[i]])
        target_deg_i = np.array([target_deg[k] for k in hits_target[i]])
        mean_source_i, mean_target_i, std_source_i, std_target_i = degree_distribution(source_deg_i, target_deg_i)
        print("Layer: {} MEAN source: {}, target: {}. STD source: {}, target: {}".format(i + 1, mean_source_i, mean_target_i, std_source_i, std_target_i))



def degree_distribution(source_deg, target_deg):
    if False:
        for i in range(len(source_deg)):
            print("Source degree: {}, target degree: {}".format(source_deg[i], target_deg[i]))
    
    
    mean_source_deg = np.mean(source_deg)
    mean_target_deg = np.mean(target_deg)
    std_source_deg = np.std(source_deg)
    std_target_deg = np.std(target_deg)

    return mean_source_deg, mean_target_deg, std_source_deg, std_target_deg
    

def get_nn_avg_dist(simi, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    # bs = 1024
    # all_distances = []
    # emb = emb.transpose(0, 1).contiguous()
    # for i in range(0, query.shape[0], bs):
    #     distances = query[i:i + bs].mm(emb) # 2014 x emb_dim * emb_dim x dim1
    #     best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
    #     all_distances.append(best_distances.mean(1).cpu())
    # all_distances = torch.cat(all_distances)

    best_simi_indice = np.argpartition(simi, -knn)[:, -knn:]
    best_simi_value = np.array([simi[i, best_simi_indice[i]] for i in range(len(best_simi_indice))]).mean(axis=1).reshape(len(best_simi_indice), 1)

    return best_simi_value




def get_candidates(simi):
    """
    Get best translation pairs candidates.
    """
    knn = '10'
    assert knn.isdigit()
    knn = int(knn)
    average_dist1 = get_nn_avg_dist(simi, knn)
    average_dist2 = get_nn_avg_dist(simi.T, knn)
    score = 2 * simi
    score = score - average_dist1 - average_dist2
    return score

# def build_dictionary(src_emb, tgt_emb, s2t_candidates=None, t2s_candidates=None, p_keep=1):
#     """
#     Build a training dictionary given current embeddings / mapping.
#     """
#     s2t = True
#     t2s = False
#     assert s2t or t2s

#     if s2t:
#         if s2t_candidates is None:
#             s2t_candidates = get_candidates(src_emb, tgt_emb, p_keep)
#     if t2s:
#         if t2s_candidates is None:
#             t2s_candidates = get_candidates(tgt_emb, src_emb, p_keep)
#         t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

#     # if params.dico_build == 'S2T':
#     dico = s2t_candidates
#     # logger.info('New train dictionary of %i pairs.' % dico.size(0))
#     return dico.cuda()

def get_L_orbits(Adj):
    g = AdjM_2_nx(Adj)
    orbits = nx_2_orbitMatrix(g)
    return orbits

def edge_list_2_nx(edge_list):
    '''
    :param edge_list: edgelist of network
    :return: a network.Graph() instance
    '''
    g = nx.Graph()
    for i, j in edge_list:
        g.add_edge(i,j)
    return g

def AdjM_2_nx(AdjM):
    g = nx.Graph()
    for row in range(len(AdjM)):
        for col in range(len(row)):
            g.add_edge(row, col)
    return g


def nx_2_orbitMatrix(g):
    num_node = g.number_of_nodes()

    ob_0 = nx.adj_matrix(g).todense()
    ob_1, ob_2, ob_3, open_triangle_set, close_triangle_set = orbit_3nodes(g)
    #ob_4, ob_5, ob_6, ob_7, ob_8, ob_9, ob_10, ob_11, ob_12, ob_13, ob_14 \
    #    = orbit_4nodes(g, open_triangle_set, close_triangle_set)
    orbits = torch.Tensor([ob_0, ob_1, ob_2, ob_3])
    return orbits

def orbit_3nodes(g):
    num_node = g.number_of_nodes()
    open_triangle_set = []
    close_triangle_set = []
    orbit_1 = np.zeros((num_node, num_node))
    orbit_2 = np.zeros((num_node, num_node))
    orbit_3 = np.zeros((num_node, num_node))
    for i in g.nodes():
        for j in g.neighbors(i):
            for k in g.neighbors(j):
                if k == i:
                    continue
                if g.has_edge(i, k) and {i, j, k} not in close_triangle_set:
                    close_triangle_set.append({i, j, k})
                    orbit_3[i, j] += 1
                    orbit_3[j, i] += 1
                    orbit_3[i, k] += 1
                    orbit_3[k, i] += 1
                    orbit_3[j, k] += 1
                    orbit_3[k, j] += 1
                elif (not g.has_edge(i, k)) and ({i, j, k} not in open_triangle_set):
                    open_triangle_set.append({i, j, k})
                    orbit_1[i, j] += 1
                    orbit_2[j, i] += 1
                    orbit_2[j, k] += 1
                    orbit_1[k, j] += 1
    return orbit_1, orbit_2, orbit_3, open_triangle_set, close_triangle_set

def orbit_4nodes(g, open_triangle_set, close_triangle_set):
    num_node = g.number_of_nodes()
    four_path_set = []
    three_star_set = []
    four_cycle_set = []
    four_tailed_triangle_set = []
    four_chordal_cycle_set = []
    four_clique_set = []

    orbit_4 = np.zeros((num_node, num_node))
    orbit_5 = np.zeros((num_node, num_node))
    orbit_6 = np.zeros((num_node, num_node))
    orbit_7 = np.zeros((num_node, num_node))
    orbit_8 = np.zeros((num_node, num_node))
    orbit_9 = np.zeros((num_node, num_node))
    orbit_10 = np.zeros((num_node, num_node))
    orbit_11 = np.zeros((num_node, num_node))
    orbit_12 = np.zeros((num_node, num_node))
    orbit_13 = np.zeros((num_node, num_node))
    orbit_14 = np.zeros((num_node, num_node))

    for threenodes in open_triangle_set:
        lst_3nodes = list(threenodes)
        # 2在中间
        if not g.has_edge(lst_3nodes[0], lst_3nodes[1]):
            a = lst_3nodes[0]
            b = lst_3nodes[2]
            c = lst_3nodes[1]
        # 1在中间
        elif not g.has_edge(lst_3nodes[0], lst_3nodes[2]):
            a = lst_3nodes[0]
            b = lst_3nodes[1]
            c = lst_3nodes[2]
        # 0在中间
        else:
            a = lst_3nodes[1]
            b = lst_3nodes[0]
            c = lst_3nodes[2]
        for n in g.neighbors(a):
            if n == b or n == c:
                continue
            edge_nb = g.has_edge(n,b)
            edge_nc = g.has_edge(n,c)
            # four_path
            if not edge_nb and not edge_nc:
                if {n, a, b, c} not in four_path_set:
                    four_path_set.append({n, a, b, c})
                    orbit_4[n, a] += 1
                    orbit_4[c, b] += 1
                    orbit_5[a, n] += 1
                    orbit_5[a, b] += 1
                    orbit_5[b, a] += 1
                    orbit_5[b, c] += 1
            elif edge_nb and not edge_nc:
                if {n, a, b, c} not in four_tailed_triangle_set:
                    four_tailed_triangle_set.append({n, a, b, c})
                    orbit_9[c, b] += 1
                    orbit_10[a, n] += 1
                    orbit_10[a, b] += 1
                    orbit_10[n, a] += 1
                    orbit_10[n, b] += 1
                    orbit_11[b, a] += 1
                    orbit_11[b, n] += 1
                    orbit_11[n, c] += 1
            elif not edge_nb and edge_nc:
                if {n, a, b, c} not in four_cycle_set:
                    four_cycle_set.append({n, a, b, c})
                    orbit_8[a, b] += 1
                    orbit_8[a, n] += 1
                    orbit_8[b, a] += 1
                    orbit_8[b, c] += 1
                    orbit_8[c, b] += 1
                    orbit_8[c, n] += 1
                    orbit_8[n, c] += 1
                    orbit_8[n, a] += 1
            elif edge_nb and edge_nc:
                if {n, a, b, c} not in four_chordal_cycle_set:
                    four_chordal_cycle_set.append({n, a, b, c})
                    orbit_12[a, b] += 1
                    orbit_12[a, n] += 1
                    orbit_12[c, b] += 1
                    orbit_12[c, n] += 1
                    orbit_13[b, a] += 1
                    orbit_13[b, c] += 1
                    orbit_13[b, n] += 1
                    orbit_13[n, a] += 1
                    orbit_13[n, b] += 1
                    orbit_13[n, c] += 1
        for n in g.neighbors(b):
            if n == a or n == c:
                continue
            if not g.has_edge(a, n) and not g.has_edge(c, n):
                if {n, a, b, c} not in three_star_set:
                    three_star_set.append({n, a, b, c})
                    orbit_6[a, b] += 1
                    orbit_6[c, b] += 1
                    orbit_6[n, b] += 1
                    orbit_7[b, a] += 1
                    orbit_7[b, c] += 1
                    orbit_7[b, n] += 1
        for n in g.neighbors(c):
            if n == b or n == a:
                continue
            edge_nb = g.has_edge(n,b)
            edge_nc = g.has_edge(n,a)
            # four_path
            if not edge_nb and not edge_nc:
                if {n, a, b, c} not in four_path_set:
                    four_path_set.append({n, a, b, c})
                    orbit_4[n, a] += 1
                    orbit_4[c, b] += 1
                    orbit_5[c, n] += 1
                    orbit_5[c, b] += 1
                    orbit_5[b, c] += 1
                    orbit_5[b, a] += 1
            elif edge_nb and not edge_nc:
                if {n, a, b, c} not in four_tailed_triangle_set:
                    four_tailed_triangle_set.append({n, a, b, c})
                    orbit_9[a, b] += 1
                    orbit_10[c, n] += 1
                    orbit_10[c, b] += 1
                    orbit_10[n, c] += 1
                    orbit_10[n, b] += 1
                    orbit_11[b, c] += 1
                    orbit_11[b, n] += 1
                    orbit_11[n, a] += 1
            elif not edge_nb and edge_nc:
                if {n, a, b, c} not in four_cycle_set:
                    four_cycle_set.append({n, a, b, c})
                    orbit_8[c, b] += 1
                    orbit_8[c, n] += 1
                    orbit_8[b, c] += 1
                    orbit_8[b, a] += 1
                    orbit_8[a, b] += 1
                    orbit_8[a, n] += 1
                    orbit_8[n, a] += 1
                    orbit_8[n, c] += 1
            elif edge_nb and edge_nc:
                if {n, a, b, c} not in four_chordal_cycle_set:
                    four_chordal_cycle_set.append({n, a, b, c})
                    orbit_12[c, b] += 1
                    orbit_12[c, n] += 1
                    orbit_12[a, b] += 1
                    orbit_12[a, n] += 1
                    orbit_13[b, c] += 1
                    orbit_13[b, a] += 1
                    orbit_13[b, n] += 1
                    orbit_13[n, c] += 1
                    orbit_13[n, b] += 1
                    orbit_13[n, a] += 1

    for threenodes in close_triangle_set:
        lst_3nodes = list(threenodes)
        a, b, c = lst_3nodes[:]
        for a in lst_3nodes:
            b, c = [x for x in lst_3nodes if x != a]
            for n in g.neighbors(a):
                if n == b or n == c:
                    continue
                edge_nb = g.has_edge(n,b)
                edge_nc = g.has_edge(n,c)
                if not edge_nb and not edge_nc:
                    if {n, a, b, c} not in four_tailed_triangle_set:
                        four_tailed_triangle_set.append({n, a, b, c})
                        orbit_9[n, a] += 1
                        orbit_10[c, a] += 1
                        orbit_10[c, b] += 1
                        orbit_10[b, c] += 1
                        orbit_10[b, a] += 1
                        orbit_11[a, c] += 1
                        orbit_11[a, n] += 1
                        orbit_11[a, b] += 1
                elif not edge_nb and edge_nc:
                    if {n, a, b, c} not in four_chordal_cycle_set:
                        four_chordal_cycle_set.append({n, a, b, c})
                        orbit_12[b, c] += 1
                        orbit_12[b, a] += 1
                        orbit_12[n, c] += 1
                        orbit_12[n, a] += 1
                        orbit_13[c, b] += 1
                        orbit_13[c, a] += 1
                        orbit_13[c, n] += 1
                        orbit_13[a, c] += 1
                        orbit_13[a, b] += 1
                        orbit_13[a, n] += 1
                elif edge_nb and edge_nc:
                    if {n, a, b, c} not in four_clique_set:
                        four_clique_set.append({n, a, b, c})
                        orbit_14[a, b] += 1
                        orbit_14[a, c] += 1
                        orbit_14[a, n] += 1
                        orbit_14[b, c] += 1
                        orbit_14[b, n] += 1
                        orbit_14[b, a] += 1
                        orbit_14[c, a] += 1
                        orbit_14[c, b] += 1
                        orbit_14[c, n] += 1
                        orbit_14[n, a] += 1
                        orbit_14[n, b] += 1
                        orbit_14[n, c] += 1
    return orbit_4, orbit_5, orbit_6, orbit_7, orbit_8, orbit_9, orbit_10, orbit_11, orbit_12, orbit_13, orbit_14

if __name__ == "__main__":

    G = nx.fast_gnp_random_graph(100, 0.4, seed=0)
    print(G.number_of_edges())
    # G = nx.barabasi_albert_graph(100,  3, seed=0)
    orbits = nx_2_orbitMatrix(G)
    for ix, orbit in enumerate(orbits):
        print('orbit', ix)
        print(orbit)
    nx.draw(G, with_labels = True)
    plt.show()




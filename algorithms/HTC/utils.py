from scipy.sparse import csr_matrix, coo_matrix
import torch
import torch.nn.functional as F
import numpy as np
import random
import networkx as nx
import heapq
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats
import pandas as pd


def load_gt(path, id2idx_src=None, id2idx_trg=None, format='matrix'):
    if id2idx_src:
        conversion_src = type(list(id2idx_src.keys())[0])
        conversion_trg = type(list(id2idx_trg.keys())[0])
    if format == 'matrix':
        # Dense
        """
        gt = np.zeros((len(id2idx_src.keys()), len(id2idx_trg.keys())))
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()                
                gt[id2idx_src[conversion_src(src)], id2idx_trg[conversion_trg(trg)]] = 1
        return gt
        """
        # Sparse
        row = []
        col = []
        val = []
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()
                row.append(id2idx_src[conversion_src(src)])
                col.append(id2idx_trg[conversion_trg(trg)])
                val.append(1)
        gt = csr_matrix((val, (row, col)), shape=(len(id2idx_src), len(id2idx_trg)))
    else:
        gt = {}
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()
                # print(src, trg)
                if id2idx_src:
                    gt[id2idx_src[conversion_src(src)]] = id2idx_trg[conversion_trg(trg)]
                else:
                    gt[str(src)] = str(trg)
    return gt


def get_elements(source_dataset, target_dataset):
    """
    Compute Adj matrix
    Preprocessing nodes attribute
    """
    source_A = source_dataset.get_adjacency_matrix()
    target_A = target_dataset.get_adjacency_matrix()

    source_feats = source_dataset.features
    target_feats = target_dataset.features
    #if no features
    if source_feats is None:
        source_feats = np.zeros((len(source_dataset.G.nodes()), 1))
        target_feats = np.zeros((len(target_dataset.G.nodes()), 1))
    #if feature row is all-zero
    for i in range(len(source_feats)):
        if source_feats[i].sum() == 0:
            source_feats[i, -1] = 1
    for i in range(len(target_feats)):
        if target_feats[i].sum() == 0:
            target_feats[i, -1] = 1

    if source_feats is not None:
        source_feats = torch.FloatTensor(source_feats)
        target_feats = torch.FloatTensor(target_feats)

    source_feats = F.normalize(source_feats)
    target_feats = F.normalize(target_feats)
    return source_A, target_A, source_feats, target_feats

def diff_mat(Adj, l):
    diff_mat = []
    adj_i = Adj = np.array(Adj)
    diff_mat.append(adj_i)
    print(type(adj_i))
    alpha = 0.15
    for i in range(1, l):
        adj_i = alpha * (1- alpha)**i*np.dot(adj_i, Adj)
        diff_mat.append(adj_i)

    diff_mat = torch.Tensor(diff_mat)
    diff_mat = diff_mat.reshape((l, Adj.shape[0], Adj.shape[1]))

    return diff_mat

def orca2gom(data_path, adj):
    edge_dir = data_path + '/orca_in.txt'
    orca_dir = data_path + '/orca_out.txt'
    edge = np.loadtxt(edge_dir)[1:,:]
    orbit_counts = np.loadtxt(orca_dir)
    num_node = adj.shape[0]
    goms = []
    goms.append(coo_matrix(adj))
    for k in range(orbit_counts.shape[1]):
        row = np.concatenate((edge[:,0], edge[:,1]))
        col = np.concatenate((edge[:,1], edge[:,0]))
        val = np.concatenate((orbit_counts[:,k], orbit_counts[:,k]))
        sp_matrix = coo_matrix((val, (row, col)), shape=(adj.shape[0], adj.shape[1]))
        goms.append(sp_matrix)
    return goms

def gom2lap(goms):
    laps = []
    for i in range(len(goms)):
        gom = goms[i].todense()
        diag = np.clip(gom.max(axis=1), a_min = 1, a_max= None)
        np.fill_diagonal(gom, diag)
        D_normed = np.diag(np.array(gom.sum(axis= 1)).reshape(-1)**-0.5)
        lap = D_normed.dot(gom).dot(D_normed)
        laps.append(lap)

    return np.array(laps)

def target_generate(noise_level, dataset, type_aug='remove_edges'):
    """
    Generate small noisy graph from original graph
    :params dataset: original graph
    :params type_aug: type of noise added for generating new graph
    """
    edges = dataset.get_edges()
    adj = dataset.get_adjacency_matrix()
    if type_aug == "remove_edges":
        num_edges = len(edges)
        num_remove = int(len(edges) * noise_level)
        for i in range(num_remove):
            index = np.random.choice(np.arange(num_edges), 1)
            row = edges[index, 0]
            col = edges[index, 1]
            while adj[row,:].sum() == 1 or adj[:,col].sum() == 1:
                index = np.random.choice(np.arange(num_edges), 1)
                row = edges[index, 0]
                col = edges[index, 1]
            adj[row, col] = adj[col,row] = 0

    elif type_aug == "add_edges":
        num_edges = len(edges)
        num_add = int(len(edges) * noise_level)
        count_add = 0
        while count_add < num_add:
            random_index = np.random.randint(0, adj.shape[1], 2)
            if adj[random_index[0], random_index[1]] == 0:
                adj[random_index[0], random_index[1]] = 1
                adj[random_index[1], random_index[0]] = 1
                count_add += 1
    elif type_aug == "change_feats":
        feats = np.copy(dataset.features)
        num_nodes = adj.shape[0]
        num_nodes_change_feats = int(num_nodes * noise_level)
        node_to_change_feats = np.random.choice(np.arange(0, adj.shape[0]), num_nodes_change_feats, replace=False)
        for node in node_to_change_feats:
            feat_node = feats[node]
            feat_node[feat_node == 1] = 0
            feat_node[np.random.randint(0, feats.shape[1], 1)[0]] = 1
        feats = torch.FloatTensor(feats)
        return feats
    return adj

def avg_top_k(M, k):
    m,_ = torch.sort(M, dim=1)
    temp = torch.mean(m[:, -k:], dim =1)
    return temp

def cos_sim( z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def CSLS(Hs, Ht, k):
    Hs_mean = torch.mean(Hs, dim = 1).reshape(-1,1)
    Ht_mean = torch.mean(Ht, dim = 1).reshape(-1,1)
    cos = cos_sim(Hs - Hs_mean, Ht - Ht_mean) # ~pearson
    r_s = avg_top_k(cos, k).reshape((-1, 1))
    r_t = avg_top_k(cos.T, k).reshape((1, -1))
    csls = 2 * cos - r_s - r_t
    return csls

def greedy_match(S):
    """
    :param S: Scores matrix, shape MxN where M is the number of source nodes,
        N is the number of target nodes.
    :return: A dict, map from source to list of targets.
    """
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    min_size = min([m, n])
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))
    col = np.zeros((min_size))

    ix = np.argsort(-x) + 1

    matched = 1
    index = 1
    while(matched <= min_size):
        ipos = ix[index-1]
        jc = int(np.ceil(ipos/m))
        ic = ipos - (jc-1)*m
        if ic == 0 : ic = 1
        if (used_rows[ic-1] == 0 and used_cols[jc-1] == 0):
            row[matched-1] = ic - 1
            col[matched-1] = jc - 1
            max_list[matched-1] = x[index-1]
            used_rows[ic-1] = 1
            used_cols[jc-1] = 1
            matched += 1
        index += 1

    result = np.zeros(S.T.shape)
    for i in range(len(row)):
        result[int(col[i]), int(row[i])] = 1
    return result

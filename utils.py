import numpy as np
import torch
import random
import torch.nn.functional as F
import math
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


# def get_neighbors(adj):
#     num_nodes = adj.shape[0]
#
#     row_indices, col_indices, _ = find(adj.A)
#     # 找到每个节点的邻居
#     neighbors = {}
#     for i in range(num_nodes):
#         # 找到第 i 个节点的邻居索引
#         neighbor_indices = col_indices[row_indices == i]
#         # 排除自身节点
#         neighbor_indices = neighbor_indices[neighbor_indices != i]
#         # 将邻居索引列表存储到邻居字典中
#         neighbors[i] = set(neighbor_indices)
#
#     return neighbors


def get_neighbors(edge_index, num_nodes):
    row0, row1 = edge_index
    neighbors = {}
    for node in range(num_nodes):
        indices = torch.where(row0 == node)[0]
        node_neighbors = row1[indices]
        neighbors[node] = set(node_neighbors.numpy())

    return neighbors


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_mis_rate(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    return ((output != labels.argmax(1)).sum()).item() / len(labels)


def get_attach_nodes(idx_others, trojan_num):
    idx_attach = np.random.choice(idx_others, size=trojan_num, replace=False)
    return idx_attach


def get_split(data, target_class, seed):
    rs = np.random.RandomState(seed)
    perm = rs.permutation(data.num_nodes)

    train_num = int(0.2 * len(perm))
    idx_train = torch.tensor(sorted(perm[:train_num]), dtype=torch.long)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True

    val_num = int(0.1 * len(perm))
    tv = train_num + val_num
    idx_val = torch.tensor(sorted(perm[train_num:tv]), dtype=torch.long)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True

    test_num = int(0.4 * len(perm))
    tvt = train_num + val_num + test_num
    idx_test = torch.tensor(sorted(perm[tv:tvt]), dtype=torch.long)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True

    idx_others = perm[tvt:]
    idx_others = idx_others[data.y[idx_others] != target_class]
    # atk_num = int(0.05 * len(perm))
    atk_num = int(0.2 * len(perm))
    idx_attach = torch.tensor(sorted(idx_others[:atk_num]), dtype=torch.long)

    return data, idx_train, idx_val, idx_test, idx_attach


def get_train_val_test(num_nodes, seed):
    rs = np.random.RandomState(seed)
    perm = rs.permutation(num_nodes)

    train_num = int(0.2 * len(perm))
    idx_train = torch.tensor(sorted(perm[:train_num]), dtype=torch.long)

    val_num = int(0.1 * len(perm))
    tv = train_num + val_num
    idx_val = torch.tensor(sorted(perm[train_num:tv]), dtype=torch.long)

    # test_num = int(0.7 * len(perm))
    # tvt = train_num + val_num + test_num
    idx_test = torch.tensor(sorted(perm[tv:]), dtype=torch.long)

    return idx_train, idx_val, idx_test


def index_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def cal_cos_sim(adj, x, edge_index, idx_attach):
    neighbors = get_neighbors(adj)
    cos_sim_list = []
    for idx in idx_attach:
        xu = x[idx]
        neighs = torch.tensor(list(neighbors[int(idx)]))
        du = math.sqrt(len(neighs))
        ru = torch.zeros(x.shape[1], dtype=torch.float)
        for j in neighs:
            neighs_j = neighbors[int(j)]
            dj = math.sqrt(len(neighs_j))
            xj = x[j]
            ru = ru + (1 / (du * dj) * xj)
        cos_sim = F.cosine_similarity(xu.unsqueeze(0), ru.unsqueeze(0))
        cos_sim_list.append(cos_sim.item())

    print(cos_sim_list)


def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    return nodes_to_keep


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels

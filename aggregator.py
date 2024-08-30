import torch
import torch.nn as nn
import random


class MeanAggregator(nn.Module):
    def __init__(self, features, num_sample=5, device='cpu'):
        super(MeanAggregator, self).__init__()

        self.features = features
        self.num_sample = num_sample
        self.device = device

    def forward(self, nodes, to_neighs):
        samp_neighs = [set(random.sample(neigh, self.num_sample)) if len(neigh) > self.num_sample else neigh
                       for neigh in to_neighs]
        samp_neighs = [samp_neigh.union({nodes[i]}) for i, samp_neigh in enumerate(samp_neighs)]

        num_nodes = len(samp_neighs)
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = torch.zeros(num_nodes, len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(num_nodes) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1

        mask = mask.to(self.device)

        # 按行求和
        num_neigh = mask.sum(1, keepdim=True)
        # 对应行元素作除法
        mask = mask.div(num_neigh)

        unique_nodes_list = torch.LongTensor(unique_nodes_list)
        if torch.is_tensor(self.features):
            embed_matrix = self.features[unique_nodes_list]
        else:
            embed_matrix = self.features(unique_nodes_list)
        # if self.device != 'cpu':
        #     unique_nodes_list.to('cpu')

        to_feats = mask.mm(embed_matrix)
        return to_feats

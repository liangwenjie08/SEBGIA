import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear


class Generator(nn.Module):
    def __init__(self, feat_dim, k_hop=2, device='cpu'):
        super(Generator, self).__init__()

        self.device = device
        self.k_hop = k_hop

        self.lins = nn.ModuleList()
        for i in range(k_hop):
            self.lins.append(Linear(feat_dim, feat_dim, bias=False).to(self.device))

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def aggregate(self, x, edge_index, size):
        mask = torch.zeros(size, dtype=torch.long, device=self.device)
        mask[edge_index[0], edge_index[1]] = 1
        mask = mask.t()
        # Sum by row
        num_neigh = mask.sum(1, keepdim=True)
        # Dividing corresponding row elements
        mask = mask.div(num_neigh)

        return mask.mm(x)

    def forward(self, x, adjs):
        if self.k_hop == 1:
            adjs = [adjs]

        for i, adj in enumerate(adjs):
            (edge_index, _, size) = adj
            embedding = self.aggregate(x, edge_index, size)
            x = F.relu(self.lins[i](embedding))

        return x

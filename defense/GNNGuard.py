import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
import torch.nn.functional as F
from torch_scatter.scatter import scatter_add
from torch_sparse import SparseTensor
import torch.optim as optim
from deeprobust.graph.utils import accuracy


class EGCNGuard(nn.Module):
    """
    Efficient GCNGuard

    """

    def __init__(self, nfeat, nhid, nclass, num_layers, idx_train, idx_val, idx_test, dropout,
                 device='cpu', lr=0.005, weight_decay=5e-4, layer_norm_first=True, use_ln=False,
                 attention_drop=True, verbose=False, threshold=0.1):
        super(EGCNGuard, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, add_self_loops=False))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(nhid, nhid, add_self_loops=False))
            self.lns.append(torch.nn.LayerNorm(nhid))
        self.lns.append(torch.nn.LayerNorm(nhid))
        self.convs.append(GCNConv(nhid, nclass, add_self_loops=False))

        self.dropout = dropout
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

        # specific designs from GNNGuard
        self.attention_drop = attention_drop
        # the definition of p0 is confusing comparing the paper and the issue
        # self.p0 = p0
        # https://github.com/mims-harvard/GNNGuard/issues/4
        self.gate = 0.  # Parameter(torch.rand(1))
        self.prune_edge = True
        self.threshold = threshold
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.verbose = verbose
        self.name = 'EGCNGuard'
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.edge_index = None
        self.x = None
        self.y = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, x, adj):
        if self.layer_norm_first:
            x = self.lns[0](x)
        new_adj = adj
        for i, conv in enumerate(self.convs[:-1]):
            new_adj = self.att_coef(x, new_adj)
            x = conv(x, new_adj)
            if self.use_ln:
                x = self.lns[i + 1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        new_adj = self.att_coef(x, new_adj)
        x = conv(x, new_adj)
        return x.log_softmax(dim=-1)

    def att_coef(self, features, adj):
        with torch.no_grad():
            row, col = adj.coo()[:2]
            n_total = features.size(0)
            if features.size(1) > 512 or row.size(0) > 5e5:
                # an alternative solution to calculate cosine_sim
                # feat_norm = F.normalize(features,p=2)
                batch_size = int(1e8 // features.size(1))
                bepoch = row.size(0) // batch_size + (row.size(0) % batch_size > 0)
                sims = []
                for i in range(bepoch):
                    st = i * batch_size
                    ed = min((i + 1) * batch_size, row.size(0))
                    sims.append(F.cosine_similarity(features[row[st:ed]], features[col[st:ed]]))
                sims = torch.cat(sims, dim=0)
                # sims = [F.cosine_similarity(features[u.item()].unsqueeze(0), features[v.item()].unsqueeze(0)).item() for (u, v) in zip(row, col)]
                # sims = torch.FloatTensor(sims).to(features.device)
            else:
                sims = F.cosine_similarity(features[row], features[col])
            mask = torch.logical_or(sims >= self.threshold, row == col)
            row = row[mask]
            col = col[mask]
            sims = sims[mask]
            has_self_loop = (row == col).sum().item()
            if has_self_loop:
                sims[row == col] = 0

            # normalize sims
            deg = scatter_add(sims, row, dim=0, dim_size=n_total)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            sims = deg_inv_sqrt[row] * sims * deg_inv_sqrt[col]

            # add self-loops
            deg_new = scatter_add(torch.ones(sims.size(), device=sims.device), col, dim=0, dim_size=n_total) + 1
            deg_inv_sqrt_new = deg_new.float().pow_(-1.0)
            deg_inv_sqrt_new.masked_fill_(deg_inv_sqrt == float('inf'), 0)

            if has_self_loop == 0:
                new_idx = torch.arange(n_total, device=row.device)
                row = torch.cat((row, new_idx), dim=0)
                col = torch.cat((col, new_idx), dim=0)
                sims = torch.cat((sims, deg_inv_sqrt_new), dim=0)
            elif has_self_loop < n_total:
                if self.verbose:
                    print(f"add {n_total - has_self_loop} remaining self-loops")
                new_idx = torch.ones(n_total, device=row.device).bool()
                new_idx[row[row == col]] = False
                new_idx = torch.nonzero(new_idx, as_tuple=True)[0]
                row = torch.cat((row, new_idx), dim=0)
                col = torch.cat((col, new_idx), dim=0)
                sims = torch.cat((sims, deg_inv_sqrt_new[new_idx]), dim=0)
                sims[row == col] = deg_inv_sqrt_new
            else:
                # print(has_self_loop)
                # print((row==col).sum())
                # print(deg_inv_sqrt_new.size())
                sims[row == col] = deg_inv_sqrt_new
            sims = sims.exp()
            graph_size = torch.Size((n_total, n_total))
            new_adj = SparseTensor(row=row, col=col, value=sims, sparse_sizes=graph_size)
        return new_adj

    def fit(self, edge_index, x, y, train_iters=500, initialize=True, **kwargs):
        if initialize:
            self.reset_parameters()

        self.edge_index = edge_index.to(self.device)
        self.x = x.to(self.device)
        self.y = y.to(self.device)

        print(f'=== training {self.name} model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.y
        idx_train, idx_val = self.idx_train, self.idx_val
        idx_combined = torch.cat((idx_train, idx_val))

        x, edge_index = self.x, self.edge_index
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(x, edge_index)
            loss_train = F.nll_loss(output[idx_combined], labels[idx_combined])
            loss_train.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i + 1, loss_train.item()))

    def test(self):
        """Evaluate model performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        idx_test = self.idx_test
        labels = self.y
        output = self.forward(self.x, self.edge_index)
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self, x=None, edge_index=None):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities)
        """
        self.eval()
        if x is None or edge_index is None:
            x, edge_index = self.x, self.edge_index
        return self.forward(x, edge_index)

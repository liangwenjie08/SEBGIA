import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from deeprobust.graph.utils import accuracy, normalize_sparse_tensor


class FLAG(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_layers, idx_train, idx_val, idx_test, step_size, m,
                 dropout, layer_norm_first=True, use_ln=False, weight_decay=5e-4, lr=0.005, gnn_epochs=500,
                 device='cpu'):
        super(FLAG, self).__init__()
        self.step_size = step_size
        self.m = m
        self.gnn_epochs = gnn_epochs
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.adj = None
        self.x = None
        self.y = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.model = GCN(nfeat, nhid, nclass, num_layers, dropout,
                         layer_norm_first=layer_norm_first, use_ln=use_ln).to(device)

    def forward(self, x, adj):
        return self.model(x, adj)

    def get_adj(self, edge_index, n):
        adj = torch.sparse_coo_tensor(edge_index, values=torch.ones(edge_index.shape[1], device=edge_index.device),
                                      size=(n, n))
        adj = normalize_sparse_tensor(adj)
        return adj

    def fit(self, edge_index, x, y, initialize=True):
        if initialize:
            self.model.reset_parameters()

        adj = self.get_adj(edge_index, x.shape[0])

        self.adj = adj.to(self.device)
        self.x = x.to(self.device)
        self.y = y.to(self.device)

        idx_train, idx_val = self.idx_train, self.idx_val
        idx_combined = torch.cat((idx_train, idx_val))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(self.gnn_epochs):
            loss = self.train_flag(self.model, self.x, self.adj, self.y, idx_combined, optimizer,
                                   self.device, self.step_size, self.m)

            if (i + 1) % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i + 1, loss))

        self.model.eval()

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
        output = self.forward(self.x, self.adj)
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
            x, adj = self.x, self.adj
        else:
            edge_index = edge_index.to('cpu')
            adj = self.get_adj(edge_index, x.shape[0])
            # adj = self.get_adj(edge_index, x.shape[0])
            adj = adj.to(self.device)

        return self.forward(x, adj)

    def train_flag(self, model, x, adj, y, idx_train, optimizer, device, step_size, m):

        y = y[idx_train]
        model_forward = (model, lambda x_: model(x_, adj)[idx_train])
        loss, _ = self.flag(model_forward, x, y, step_size, m, optimizer, device, F.nll_loss)

        return loss.item()

    def flag(self, model_forward, clean, y, step_size, m, optimizer, device, criterion):
        model, forward = model_forward
        model.train()
        optimizer.zero_grad()

        perturb = torch.FloatTensor(*clean.shape).uniform_(-step_size, step_size).to(device)
        perturb.requires_grad_()
        out = forward(perturb + clean)
        loss = criterion(out, y)
        loss /= m

        for _ in range(m - 1):
            loss.backward()
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0

            out = forward(perturb + clean)
            loss = criterion(out, y)
            loss /= m

        loss.backward()
        optimizer.step()

        return loss, out


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_layers=3, dropout=0.5,
                 layer_norm_first=False, use_ln=True):
        super(GCN, self).__init__()
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, cached=False))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(nhid, nhid, cached=False))
            self.lns.append(torch.nn.LayerNorm(nhid))
        self.lns.append(torch.nn.LayerNorm(nhid))
        self.convs.append(GCNConv(nhid, nclass, cached=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, x, adj_t, layers=-1):
        if self.layer_norm_first:
            x = self.lns[0](x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.use_ln:
                x = self.lns[i + 1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # obtain output from the i-th layer
            if layers == i + 1:
                return x
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

    def con_forward(self, x, adj_t, layers=-1):
        if self.layer_norm_first and layers == 1:
            x = self.lns[0](x)
        for i in range(layers - 1, len(self.convs) - 1):
            x = self.convs[i](x, adj_t)
            if self.use_ln:
                x = self.lns[i + 1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

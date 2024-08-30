import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn.conv import GCNConv
from torch_sparse import SparseTensor
import torch.optim as optim
from copy import deepcopy
from deeprobust.graph import utils


class PGCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, nlayers=2, threshold=0.1, dropout=0.5, lr=0.01, with_relu=False,
                 layer_norm_first=True, use_ln=False, with_bn=False, weight_decay=5e-4, with_bias=True, device=None):
        super(PGCN, self).__init__()
        print(f'=== the surrogate model PGCN threshold is: {threshold} ===')

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.layers = nn.ModuleList([])
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        if with_bn:
            self.bns = nn.ModuleList()

        if nlayers == 1:
            self.layers.append(GCNConv(nfeat, nclass, bias=with_bias))
        else:
            self.layers.append(GCNConv(nfeat, nhid, bias=with_bias))
            if with_bn:
                self.bns.append(nn.BatchNorm1d(nhid))
            for i in range(nlayers - 2):
                self.layers.append(GCNConv(nhid, nhid, bias=with_bias))
                self.lns.append(torch.nn.LayerNorm(nhid))
                if with_bn:
                    self.bns.append(nn.BatchNorm1d(nhid))
            self.lns.append(torch.nn.LayerNorm(nhid))
            self.layers.append(GCNConv(nhid, nclass, bias=with_bias))

        self.dropout = dropout
        self.layer_norm_first = layer_norm_first
        self.with_relu = with_relu
        self.use_ln = use_ln
        self.weight_decay = weight_decay
        self.lr = lr
        self.threshold = threshold
        self.output = None
        self.best_model = None
        self.best_output = None

        self.x = None
        self.edge_index = None
        self.y = None
        self.edge_weight = None
        self.idx_train = None
        self.idx_val = None

        self.with_bn = with_bn
        self.name = 'PGCN'

    def forward(self, x, edge_index, edge_weight=None):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        if self.layer_norm_first:
            x = self.lns[0](x)
        for ii, layer in enumerate(self.layers):
            if edge_weight is not None:
                adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=2 * x.shape[:1]).t()
                x = layer(x, adj)
            else:
                x = layer(x, edge_index)
            if ii != len(self.layers) - 1:
                if self.with_bn:
                    x = self.bns[ii](x)
                if self.use_ln:
                    x = self.lns[ii + 1](x)

                if self.with_relu:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

    def predict(self, x=None, edge_index=None):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities)
        """
        self.eval()
        edge_weight = self.att_coef(x, edge_index)
        if x is None or edge_index is None:
            x, edge_index = self.x, self.edge_index
        return self.forward(x, edge_index, edge_weight)

    def test(self, idx_test):
        """Evaluate model performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        labels = self.y
        output = self.forward(self.x, self.edge_index, self.edge_weight)
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def fit(self, x, edge_index, y, idx_train, idx_val, train_iters=1000, initialize=True, verbose=False, patience=100,
            **kwargs):
        if initialize:
            self.initialize()
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.edge_weight = self.att_coef(x, edge_index)

        self.train_with_early_stopping(train_iters, patience, verbose)

    def att_coef(self, feat, edge_index):
        """we don't change the edge_index, just update the edge_weight;
        some edge_weight are regarded as removed if it equals to zero"""
        feat = feat.detach()
        sim = torch.cosine_similarity(feat[edge_index[0]], feat[edge_index[1]])
        sim[sim < self.threshold] = 0.0
        return sim

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print(f'=== training {self.name} model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.y
        idx_train, idx_val = self.idx_train, self.idx_val

        early_stopping = patience
        best_loss_val = 100
        best_acc_val = 0
        best_epoch = 0

        x, edge_index, edge_weight = self.x, self.edge_index, self.edge_weight
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            output = self.forward(x, edge_index, edge_weight)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 50 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(x, edge_index, edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
                best_epoch = i
            else:
                patience -= 1

            if i > early_stopping and patience <= 0:
                break

        if verbose:
            print('=== early stopping at {0}, acc_val = {1} ==='.format(best_epoch, best_acc_val))
        self.load_state_dict(weights)

    def initialize(self):
        for m in self.layers:
            m.reset_parameters()
        if self.with_bn:
            for bn in self.bns:
                bn.reset_parameters()

    def _ensure_contiguousness(self, x, edge_idx, edge_weight):
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight

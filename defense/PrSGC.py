import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SGConv
from deeprobust.graph.utils import accuracy
from copy import deepcopy


class PrSGC(nn.Module):
    def __init__(self, nfeat, nclass, K=2, threshold=0.1, lr=0.01, weight_decay=5e-4,
                 with_relu=False):
        super().__init__()
        self.threshold = threshold
        self.with_relu = with_relu
        self.lr = lr
        self.weight_decay = weight_decay

        self.ln = nn.LayerNorm(nfeat)

        self.conv1 = SGConv(
            in_channels=nfeat,
            out_channels=nclass,
            K=K,
            cached=False,
            )

        self.x = None
        self.edge_index = None
        self.edge_weight = None
        self.y = None
        self.idx_train = None
        self.idx_val = None
        self.output = None

    def forward(self, x, edge_index, edge_weight):
        x, edge_index, edge_weight = self._ensure_contiguousness(x, edge_index, edge_weight)
        x = self.ln(x)
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

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

        edge_weight = self.att_coef(x, edge_index)

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
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def fit(self, x, edge_index, y, idx_train, idx_val, train_iters=500, initialize=True, patience=100, **kwargs):
        if initialize:
            self.conv1.reset_parameters()

        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.edge_weight = self.att_coef(x, edge_index)

        self.train_with_early_stopping(train_iters, patience)

    def train_with_early_stopping(self, train_iters, patience):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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

            self.eval()
            output = self.forward(x, edge_index, edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = accuracy(output[idx_val], labels[idx_val])

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

        self.load_state_dict(weights)

    def att_coef(self, feat, edge_index):
        """we don't change the edge_index, just update the edge_weight;
        some edge_weight are regarded as removed if it equals to zero"""
        feat = feat.detach()
        sim = torch.cosine_similarity(feat[edge_index[0]], feat[edge_index[1]])
        sim[sim < self.threshold] = 0.0
        return sim

    def _ensure_contiguousness(self, x, edge_idx, edge_weight):
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight

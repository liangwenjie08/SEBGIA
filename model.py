import torch
from generator import Generator
import utils
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from defense.PrSGC import PrSGC
from torch_geometric.loader import NeighborSampler


class SEBGIA:
    def __init__(self, edge_index, x, y, idx_train, idx_val, idx_test, k_hop=2, num_sample=None, alpha=10,
                 batch_size=4096, lr=0.01, clamp_x=True, nhid=32, weight_decay=5e-4, epochs=500, surrogate_epochs=500,
                 device='cpu'):
        self.generator = None
        self.surrogate = None
        print(f'=== the alpha is: {alpha} ===')
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        if num_sample is None:
            num_sample = [3] * k_hop
        self.num_sample = num_sample
        self.k_hop = k_hop

        self.edge_index = edge_index.to(device)
        self.x = x.to(device)
        self.y = y.to(device)

        max_x = int(self.x.max().item())
        min_x = int(self.x.min().item())
        max_x = min(20, max_x)
        min_x = max(-20, min_x)
        self.max_x = max_x
        self.min_x = min_x
        self.clamp_x = clamp_x

        self.alpha = alpha
        self.lr = lr
        self.batch_size = batch_size
        self.nhid = nhid
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.surrogate_epochs = surrogate_epochs
        self.device = device

    def get_poisoned_graph(self, idx_test):
        assert self.generator, 'please first use the train() function to train feature generator'

        self.generator.eval()
        x, y = self.x, self.y
        poisoned_edge_index = self.edge_index

        x_all = []
        start = x.shape[0]
        loader = NeighborSampler(self.edge_index, node_idx=idx_test, sizes=self.num_sample,
                                 batch_size=self.batch_size, shuffle=False)
        for batch_size, n_id, adjs in loader:
            batch_nodes = n_id[:batch_size]
            batch_x = x[n_id]
            poisoned_edge_index = self.get_poisoned_edge_index(poisoned_edge_index, batch_nodes, start)

            gen_x = self.generator(batch_x, adjs).detach()
            x_all.append(gen_x)

            start += gen_x.shape[0]

        poisoned_x = torch.cat(x_all, dim=0)
        poisoned_x = poisoned_x.view([-1, x.shape[1]])
        if self.clamp_x:
            poisoned_x = torch.clamp(poisoned_x, min=self.min_x, max=self.max_x)

        poisoned_x = torch.cat([x, poisoned_x])

        poisoned_y = torch.cat([y, -1 * torch.ones([len(idx_test)], dtype=torch.long, device=self.device)])

        return poisoned_edge_index, poisoned_x, poisoned_y

    def get_poisoned_edge_index(self, edge_index, nodes, start):
        edge_list = []
        for i, idx in enumerate(nodes):
            edge_list.append([idx, i + start])
            edge_list.append([i + start, idx])
        edge_list = torch.tensor(edge_list)
        edge_list = edge_list.to(self.device)
        poisoned_edge_index = torch.cat([edge_index, edge_list.T], dim=1)

        return poisoned_edge_index

    def train_surrogate(self):
        edge_index, x, y = self.edge_index, self.x, self.y
        idx_train, idx_val, idx_test = self.idx_train, self.idx_val, self.idx_test
        self.surrogate = PrSGC(nfeat=x.shape[1], nclass=y.max().item() + 1).to(self.device)
        self.surrogate.fit(x, edge_index, y, idx_train, idx_val, train_iters=self.surrogate_epochs)
        print('=== Surrogate model test ===')
        self.surrogate.test(idx_test)
        self.surrogate.eval()

    def train(self):
        print('=== Training SBNIA model ===')
        x, y = self.x, self.y

        self.train_surrogate()

        self.generator = Generator(x.shape[1], k_hop=self.k_hop, device=self.device)
        generator_optim = optim.Adam(self.generator.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc = 10000
        weights = deepcopy(self.generator.state_dict())

        loader = NeighborSampler(self.edge_index, node_idx=self.idx_test, sizes=self.num_sample,
                                 batch_size=self.batch_size, shuffle=True)

        for i in range(self.epochs):
            total_loss = total_acc = 0

            for batch_size, n_id, adjs in loader:
                self.generator.train()
                generator_optim.zero_grad()

                batch_nodes = n_id[:batch_size]
                batch_x = x[n_id]

                poisoned_edge_index = self.get_poisoned_edge_index(self.edge_index, batch_nodes, x.shape[0])

                poisoned_x = self.generator(batch_x, adjs)
                poisoned_x = poisoned_x.view([-1, x.shape[1]])
                if self.clamp_x:
                    poisoned_x = torch.clamp(poisoned_x, min=self.min_x, max=self.max_x)
                update_x = torch.cat([x, poisoned_x])

                x_attach = x[batch_nodes].to(self.device)
                cos_sim = -torch.mean(F.cosine_similarity(x_attach, poisoned_x))

                logits = self.surrogate.predict(update_x, poisoned_edge_index)
                loss = -F.nll_loss(logits[batch_nodes], y[batch_nodes]) + self.alpha * cos_sim

                loss.backward()
                generator_optim.step()

                self.generator.eval()

                total_loss += float(loss)
                total_acc += utils.accuracy(logits[batch_nodes], y[batch_nodes])

            acc = total_acc / len(loader)
            if acc < best_acc:
                weights = deepcopy(self.generator.state_dict())
                best_acc = acc

            if (i + 1) % 100 == 0:
                print(
                    f'epoch: {i + 1}==>{self.epochs}   loss: {total_loss / len(loader)}   acc: {acc}')

        print('the best_acc is: ', best_acc)
        self.generator.load_state_dict(weights)
        self.generator.eval()

import os
import argparse
import utils
import torch
from model import SEBGIA
from defense.GNNGuard import EGCNGuard
from torch_sparse import SparseTensor


def get_sparse_tensor(edge_idx, n):
    row = edge_idx[0]
    col = edge_idx[1]
    sp_edge_idx = SparseTensor(row=row, col=col, value=torch.ones(col.size(0)), sparse_sizes=torch.Size((n, n)),
                               is_sorted=True)
    return sp_edge_idx


def get_optimal_alpha(dataset, alpha=5):
    if dataset == 'ogbarxiv':
        alpha = 5
    elif dataset == 'reddit':
        alpha = 20
    elif dataset == 'ogbproducts':
        alpha = 5

    return alpha


def run_gnnguard(dataset_name=None, k_hop=2, alpha=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='ogbproducts',
                        choices=['ogbarxiv', 'ogbproducts', 'reddit'], help='dataset')
    parser.add_argument('--gnn_epochs', type=int, default=1000, help='The traversal number of gnn model.')
    parser.add_argument('--epochs', type=int, default=1000, help='The traversal number of backdoor.')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--nhid', type=int, default=128)
    parser.add_argument('--layer_norm_first', default=True, action="store_true")
    parser.add_argument('--use_ln', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    utils.set_seed(args.seed)

    if dataset_name is not None:
        dataset = dataset_name
    else:
        dataset = args.dataset

    print(f'=== loading {dataset} dataset ===')

    sep = os.sep
    dataset_path = f'{os.getcwd()}{sep}dataset{sep}{dataset}.pt'
    data = torch.load(dataset_path)

    # Obtain the optimal alpha for each dataset
    if alpha is None:
        alpha = get_optimal_alpha(dataset)

    edge_index, x, y = data.edge_index, data.x, data.y
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # end = int(data.num_nodes * 0.05)
    # idx_test = idx_test[:end]

    sp_edge_index = get_sparse_tensor(edge_index, x.shape[0]).to(device)

    agent = SEBGIA(edge_index, x, y, idx_train, idx_val, idx_test, k_hop=k_hop, alpha=alpha,
                  lr=0.01, epochs=args.epochs, device=device)
    agent.train()

    model = EGCNGuard(x.shape[1], args.nhid, data.y.max().item() + 1, args.num_layers, idx_train, idx_val, idx_test,
                      args.dropout, device, layer_norm_first=args.layer_norm_first,
                      use_ln=args.use_ln).to(device)
    model.fit(sp_edge_index, x, y, train_iters=args.gnn_epochs)

    print('model test before attack', '=>' * 30)
    clean_acc = model.test()

    idx_test = data.idx_test
    poisoned_edge_index, poisoned_x, poisoned_y = agent.get_poisoned_graph(idx_test)

    poisoned_edge_index = poisoned_edge_index.to('cpu')
    sp_poisoned_edge_index = get_sparse_tensor(poisoned_edge_index, poisoned_x.shape[0]).to(device)

    logits = model.predict(poisoned_x, sp_poisoned_edge_index)

    print('after attack', '=>' * 30)
    acc = float(utils.accuracy(logits[idx_test], poisoned_y[idx_test]))
    mis_rate = round(1 - acc, 4)
    print(f'the accuracy after attack: {acc}')
    print(f'the misclassification rate: {mis_rate}')

    return mis_rate, clean_acc


if __name__ == '__main__':
    run_gnnguard()

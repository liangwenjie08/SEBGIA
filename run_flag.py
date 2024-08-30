import os
import torch
import argparse
import utils
from model import SEBGIA
from defense.FLAG import FLAG


def run_flag(dataset_name=None, alpha=5):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='ogbproducts',
                        choices=['ogbarxiv', 'ogbproducts', 'reddit'], help='dataset')
    parser.add_argument('--gnn_epochs', type=int, default=1000, help='The traversal number of gnn model.')
    parser.add_argument('--epochs', type=int, default=1000, help='The traversal number of backdoor.')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--layer_norm_first', default=True, action="store_true")
    parser.add_argument('--use_ln', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--step_size', type=float, default=1e-3)
    parser.add_argument('--m', type=int, default=3)

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

    edge_index, x, y = data.edge_index, data.x, data.y
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    agent = SEBGIA(edge_index, x, y, idx_train, idx_val, idx_test, alpha=alpha,
                  lr=0.01, epochs=args.epochs, device=device)
    agent.train()

    print('train test gnn model', '=>' * 30)

    model = FLAG(x.shape[1], args.nhid, y.max().item() + 1, args.num_layers, idx_train, idx_val, idx_test,
                 args.step_size, args.m, args.dropout, device=device, layer_norm_first=args.layer_norm_first,
                 use_ln=args.use_ln, gnn_epochs=args.gnn_epochs).to(device)
    model.fit(edge_index, x, y)

    print('model test before attack', '=>' * 30)
    clean_acc = model.test()

    idx_atk = data.idx_test
    poisoned_edge_index, poisoned_x, poisoned_y = agent.get_poisoned_graph(idx_atk)

    logits = model.predict(poisoned_x, poisoned_edge_index)

    print('after attack', '=>' * 30)
    acc = float(utils.accuracy(logits[idx_test], poisoned_y[idx_test]))
    mis_rate = round(1 - acc, 4)
    print(f'the accuracy after attack: {acc}')
    print(f'the misclassification rate: {mis_rate}')

    return mis_rate, clean_acc


if __name__ == '__main__':
    run_flag()

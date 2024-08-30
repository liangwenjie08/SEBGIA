from run_flag import run_flag
from run_gnnguard import run_gnnguard
import os
import pandas as pd

sep = os.sep
storage_path = f'{os.getcwd()}{sep}storage{sep}alpha.csv'

data_list = [['defense', 'dataset', 'alpha', 'mis_rate']]

defense_model = ['gnnguard', 'flag']
dataset_list = ['ogbarxiv', 'ogbproducts', 'reddit']

for defense in defense_model:
    # alpha_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for dataset in dataset_list:
        alpha_list = [0, 5, 10, 15, 20, 25]
        for alpha in alpha_list:
            if defense == 'gnnguard':
                mis_rate, _ = run_gnnguard(dataset_name=dataset, alpha=alpha)
            elif defense == 'flag':
                mis_rate, _ = run_flag(dataset_name=dataset, alpha=alpha)
            else:
                mis_rate = 0

            data_list.append([defense, dataset, alpha, mis_rate])

df = pd.DataFrame(data_list[1:], columns=data_list[0])
df.to_csv(storage_path, mode='a', index=False, sep='\t')

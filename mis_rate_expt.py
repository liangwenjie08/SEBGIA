from run_flag import run_flag
from run_gnnguard import run_gnnguard
import os
import pandas as pd

sep = os.sep
storage_path = f'{os.getcwd()}{sep}storage{sep}mis_rate.csv'

# dataset_list = ['ogbproducts']
dataset_list = ['ogbarxiv', 'ogbproducts', 'reddit']
defense_model = ['gnnguard', 'flag']

data_list = [['dataset', 'defense', 'index', 'mis_rate', 'clean_acc']]
# Number of experiments
num_expt = 1

for dataset in dataset_list:

    for defense in defense_model:
        for idx in range(num_expt):
            print(f'{dataset}--{defense}--{idx + 1}=====>{num_expt}')
            if defense == 'gnnguard':
                mis_rate, clean_acc = run_gnnguard(dataset_name=dataset)
            elif defense == 'flag':
                mis_rate, clean_acc = run_flag(dataset_name=dataset)
            else:
                mis_rate = 0
                clean_acc = 0

            data_list.append([dataset, defense, idx, mis_rate, clean_acc])

df = pd.DataFrame(data_list[1:], columns=data_list[0])
df.to_csv(storage_path, mode='a', index=False, sep='\t')

from run_gnnguard import run_gnnguard
import os
import pandas as pd

sep = os.sep
storage_path = f'{os.getcwd()}{sep}storage{sep}k_hop.csv'

data_list = [['dataset', 'defense', 'k_hop', 'mis_rate']]

dataset_list = ['ogbarxiv']
defense_model = ['gnnguard']
for dataset in dataset_list:
    for defense in defense_model:
        # k_hop_list = [1, 2, 3, 4, 5]
        k_hop_list = [5]
        for k_hop in k_hop_list:
            print('the k-hop is: ', k_hop)
            mis_rate, _ = run_gnnguard(dataset_name=dataset, k_hop=k_hop)

            data_list.append([dataset, defense, k_hop, mis_rate])

df = pd.DataFrame(data_list[1:], columns=data_list[0])
df.to_csv(storage_path, mode='a', index=False, sep='\t')

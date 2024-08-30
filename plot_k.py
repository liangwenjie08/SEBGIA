import os
import platform
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if 'windows' in platform.system().lower():
    matplotlib.use('TkAgg')

sep = os.sep
path = f'{os.getcwd()}{sep}storage{sep}k_hop.csv'
save_path = f'{os.getcwd()}{sep}imgs{sep}k_hop_ogbproducts.png'
df = pd.read_csv(path, sep='\t')
k_hop = df['k_hop']
mis_rate = df['mis_rate']

fig, ax = plt.subplots(figsize=(4.5, 4))

ax.set_ylabel('Misclassification Rate')
ax.set_xlabel('Sampling Layers')
ax.set_xticks(np.arange(1, 6))
ax.plot(k_hop, mis_rate, linewidth=0.7, marker='o', markersize=3, color='red')
ax.set_yticks(np.arange(0.5, 1, 0.1))
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
# ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

plt.savefig(save_path, dpi=500)

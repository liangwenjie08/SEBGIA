import os
import platform
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if 'windows' in platform.system().lower():
    matplotlib.use('TkAgg')

sep = os.sep
path = f'{os.getcwd()}{sep}storage{sep}alpha.csv'
save_path = f'{os.getcwd()}{sep}imgs{sep}alpha.png'
df = pd.read_csv(path, sep='\t')
left_key = 'gnnguard'
right_key = 'flag'
left_df = df[df['defense'] == left_key]
right_df = df[df['defense'] == right_key]

dataset = ['ogbn-arxiv', 'ogbn-products', 'reddit']
marks = ['o', '^', 's']

fig = plt.figure(figsize=(6, 3))

ax1 = fig.add_subplot(1, 2, 1)

ax1.set_ylabel("Misclassification Rate", fontsize=12)
ax1.set_xlabel('(a) GNNGuard', fontsize=12)
ax1.set_xticks(np.arange(0, 26, 5))
s = 0
e = 6
step = 6
for i in range(3):
    l = left_df.iloc[s:e]
    alpha = l['alpha']
    mis_rate = l['mis_rate']
    ax1.plot(alpha, mis_rate, label=dataset[i], marker=marks[i], linewidth=0.7, markersize=3)
    s += step
    e += step

ax1.set_yticks(np.arange(0, 1.1, 0.2))

ax2 = fig.add_subplot(1, 2, 2)

ax2.set_ylabel("Misclassification Rate", fontsize=12)
ax2.set_xlabel('(b) FLAG', fontsize=12)
ax2.set_xticks(np.arange(0, 26, 5))
s = 0
e = 6
for i in range(3):
    l = right_df.iloc[s:e]
    alpha = l['alpha']
    mis_rate = l['mis_rate']
    ax2.plot(alpha, mis_rate, label=dataset[i], marker=marks[i], linewidth=0.7, markersize=3)
    s += step
    e += step

ax2.set_yticks(np.arange(0, 1.1, 0.2))

lines = []
labels = []
# for ax in fig.axes:
axLine, axLabel = ax1.get_legend_handles_labels()
lines.extend(axLine)
labels.extend(axLabel)

fig.legend(lines, labels, loc='upper center', ncol=3, frameon=False, fontsize=6)

plt.tight_layout()

plt.savefig(save_path, dpi=500)

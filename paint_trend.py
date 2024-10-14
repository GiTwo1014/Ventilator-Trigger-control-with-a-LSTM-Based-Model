import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import pandas as pd

dataAll = pd.DataFrame()

fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(9, 6))

# ax.axvline(x=-33, ymin=0, ymax=100, color='gray', linestyle='-.', linewidth=0.5)
# ax.axhline(y=-33, xmin=0, xmax=100, color='gray', linestyle='-.', linewidth=0.5)


colors = np.linspace(0, 100, 10)
# 读取数据
folder_path = r'./'
count1 = 0
# data folder
i = 0
listTitle = ["Non I/E", "Inspiratory", "Expiratory"]
for ax in axs:
    data = pd.read_excel("step25-flow.xlsx")
    data = data[data.loc[:, "I/E"] == i].iloc[:, 0:25]
    ax.set_title(listTitle[count1])
    ax.set(xlim=(1, 25), ylim=(0, 1))
    ax.set_xlabel('Sample Point')
    if count1 == 0 or count1 == 1:
        ax.set_ylabel('Flow')
    else:
        ax.set_ylabel('Flow')

    # n = 100_000
    # x = np.random.standard_normal(n) * 25
    # norm = mcolors.Normalize(vmin=0, vmax=100)
    # y = (2.0 + 3.0 * x / 25 + 4.0 * np.random.standard_normal(n)) * 5]
    m, n = (data.shape)
    x = [x for x in range(1, 26)] * m
    print()
    count1 += 1
    y1 = data.values.flatten()
    y = []
    count = 0
    for i in y1:
        y.append(y1)
    print()
    ax1 = ax
    hb = ax.hexbin(x, y, gridsize=25, cmap='inferno')
    # hb = ax.hexbin(data.values, gridsize=25, cmap='')

    if count1 == 2 or count1 == 4:
        cb = fig.colorbar(hb, ax=ax, label='counts')
    else:
        cb = fig.colorbar(hb, ax=ax)
    i += 1
# plt.savefig("./trendForPIOn.png")
plt.subplots_adjust(hspace=0.3)
plt.show()
# plt.savefig("trend_flow_25.png", dpi=300)
# plt.show()

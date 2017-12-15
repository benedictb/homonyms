import numpy as np

import source.data as datalib
from matplotlib import pyplot as plt

filepath = './results/kmeans.results.2'
with open(filepath) as f:
    data = [l.strip('\n').split(',') for l in f][1:]

with open('./results/baseline.results') as f:
    bl_data = [l.strip('\n').split(',') for l in f][1:]

data = [(l[0], float(l[1]), float(l[2])) for l in data]

data_var_sorted = sorted(data, key=lambda x: datalib.get_n_clusters()[x[0]])
data_count_sorted = sorted(data, key=lambda x: datalib.get_counts()[x[0]])
# bl_data = sorted(bl_data, key=lambda x: datalib.get_counts()[[0]])

d = {l[0]: l[2] for l in bl_data}
bl_y = [d[l[0]] for l in data_count_sorted]
bl_labels = [l[0] for l in data_count_sorted]

y_var = [l[2] for l in data_var_sorted]
labels_var = [l[0] + ' ({})'.format(datalib.get_n_clusters()[l[0]]) for l in data_var_sorted]
y_count = [l[2] for l in data_count_sorted]
labels_count = [l[0] + '({0:.1f}k)'.format(datalib.get_counts()[l[0]] / float(1000)) for l in data_count_sorted]
N = len(y_var)
x_range = range(N)

#
# fig = plt.figure()
# ax = plt.subplot()
# ax.bar(x_range, y_var)
# ax.set_xticks(np.arange(N))
# ax.set_xticklabels(labels_var, rotation=45)
# ax.set_xlabel('Number of clusters (increasing)')
# ax.set_ylabel('Accuracy')
# ax.set_title('Acc vs Number of Possible Translations')
# fig.tight_layout()
#
# rects = ax.patches
#
# plt.show()
#
#
# fig = plt.figure()
# ax = plt.subplot()
# ax.bar(x_range, y_count)
# ax.set_xticks(np.arange(N))
# ax.set_xticklabels(labels_count, rotation=45)
# ax.set_xlabel('Number of Training Size (increasing)')
# ax.set_ylabel('Accuracy')
# ax.set_title('Acc vs Sample size')
# fig.tight_layout()
#
# plt.show()

fig = plt.figure()
ax = plt.subplot()
ax.bar(x_range, bl_y)
ax.set_xticks(np.arange(N))
ax.set_xticklabels(bl_labels, rotation=45)
ax.set_xlabel('Homonym')
ax.set_ylabel('Accuracy')
ax.set_title('Baseline Results')
fig.tight_layout()

plt.show()

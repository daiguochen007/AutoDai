# -*- coding: utf-8 -*-

import random as rd

import matplotlib.pyplot as plt
import numpy as np

x1 = [rd.gauss(6, 1) for i in range(100)] + [rd.gauss(0, 1) for i in range(100)] + [rd.gauss(11, 1) for i in range(100)]
x2 = [rd.gauss(6, 1) for i in range(100)] + [rd.gauss(0, 1) for i in range(100)] + [rd.gauss(1, 1) for i in range(100)]
x3 = [rd.gauss(6, 1) for i in range(100)] + [rd.gauss(0, 1) for i in range(100)] + [rd.gauss(2, 1) for i in range(100)]

data = np.transpose(np.array(x1 + x2).reshape((2, 300)))
data2 = np.transpose(np.array(x1 + x2 + x3).reshape((3, 300)))

# 2D
plt.scatter(data[:, 0], data[:, 1])
# 3D
plt.figure().add_subplot(111, projection='3d').scatter(data2[:, 0], data2[:, 1], data2[:, 2])
plt.show()


def return_cluster(sample, center_list):
    temp = [sum((sample - c) ** 2) for c in center_list]
    return temp.index(min(temp))


def init_center(data, k):
    sample_num = data.shape[0]
    return data[rd.sample(range(sample_num), k), :]


# k-means cluster
def kmeans(data, k):
    sample_num = data.shape[0]
    cluster_label = np.zeros(sample_num) - 1
    cluster_change = True
    # step 1: initiate centroids  
    centers = init_center(data, k)
    while cluster_change:
        cluster_change = False
        # step 2: caculate cluster
        for i in range(sample_num):
            new_label = return_cluster(data[i, :], centers)
            if cluster_label[i] != new_label:
                cluster_label[i] = new_label
                cluster_change = True
                # step 3: update centers
        for j in range(k):
            centers[j, :] = np.mean(data[cluster_label == j, :], axis=0)
    # caculate cost for each cluster
    cluster_costs = np.zeros(k)
    for i in range(k):
        cluster_costs[i] = np.sum((data[cluster_label == i, :] - centers[i]) ** 2)
    return centers, cluster_label, cluster_costs


def plot_cluster(data, k, centers, cluster_label):
    sample_num, dim = data.shape
    if k <= 7:
        sample_num = data.shape[0]
        # res = kmeans(data, k)
        # centers = res[0]
        # cluster_label = res[1]
        mark = ['r', 'b', 'g', 'k', 'c', 'm', 'y']
        if dim == 2:
            # draw all samples
            for i in range(sample_num):
                plt.plot(data[i, 0], data[i, 1], 'o' + mark[int(cluster_label[i])])
                # draw the centroids
            for i in range(k):
                plt.plot(centers[i, 0], centers[i, 1], '*' + mark[i], markersize=12)
            plt.show()
        elif dim == 3:
            ax = plt.figure().add_subplot(111, projection='3d')
            for i in range(sample_num):
                ax.scatter(data2[i, 0], data2[i, 1], data2[i, 2], c=mark[int(cluster_label[i])], s=12)
            # for i in range(k):
            # ax.scatter(centers[i, 0],centers[i, 1],centers[i, 2],c=mark[i],s=100,marker='*')
            plt.show()
        else:
            return False
    else:
        return False


# plot cluster by k means    
res = kmeans(data, 3)
plot_cluster(data, 3, res[0], res[1])  # 2d
res = kmeans(data2, 3)
plot_cluster(data2, 3, res[0], res[1])  # 3d

# choose k
# plot cost function value ~ numbers of k
# 2d
plt.plot(list(range(1, 7)), np.mean([[sum(kmeans(data, i)[2]) for i in range(1, 7)] for j in range(15)], axis=0))
# 3d
plt.plot(list(range(1, 7)), np.mean([[sum(kmeans(data2, i)[2]) for i in range(1, 7)] for j in range(15)], axis=0))


################# bisect K-means
def bi_kmeans(data, k):
    centers = []
    final_data = np.mat(np.zeros(data.shape[1]))
    cluster_label = []
    current_label = 0
    while current_label < k:
        if current_label == k - 1:
            # assign the rest, don't cluster again
            final_data = np.concatenate((final_data, data), axis=0)
            cluster_label = np.concatenate((cluster_label, np.zeros(len(data)) + current_label))
            current_label += 1
            centers.append(np.mean(data, axis=0))
        else:
            res = kmeans(data, 2)
            good = np.argmin(res[2])
            again = 1 - good
            final_data = np.concatenate((final_data, data[res[1] == good, :]), axis=0)
            cluster_label = np.concatenate((cluster_label, np.zeros(len(data[res[1] == good, :])) + current_label))
            current_label += 1
            centers.append(res[0][good])
            data = data[res[1] == again, :]
    final_data = np.array(final_data[1:, :])
    # caculate cost for each cluster
    cluster_costs = np.zeros(k)
    for i in range(k):
        cluster_costs[i] = np.sum((final_data[cluster_label == i, :] - centers[i]) ** 2)
    return np.array(centers), cluster_label, cluster_costs, final_data


# plot cluster by bisect k means  
res = bi_kmeans(data, 3)
plot_cluster(res[3], 3, res[0], res[1])  # 2d
res = bi_kmeans(data2, 3)
plot_cluster(res[3], 3, res[0], res[1])  # 3d

# choose k
# plot cost function value ~ numbers of k
# 2d
plt.plot(list(range(1, 7)), [sum(bi_kmeans(data, i)[2]) for i in range(1, 7)])
# 3d
plt.plot(list(range(1, 7)), [sum(bi_kmeans(data2, i)[2]) for i in range(1, 7)])

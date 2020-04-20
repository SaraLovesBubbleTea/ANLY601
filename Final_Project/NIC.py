import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.core._multiarray_umath import ndarray
from scipy.spatial.distance import pdist
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from multiprocessing import Pool
from scipy.cluster.vq import whiten

class NIC:

    def __init__(self, c=2):

        self.C = c
        self.labels = []
        self.l_label = [w for w in range(c)]
        self.scores = np.zeros(c)
        self.X = None
        self.over_all_score = 0
        self.itr = 100

    @staticmethod
    def get_index(target, label_list):
        """
        function get position index of target cluster at label list
        :param label_list: a list contains all labels
        :param target: target cluster
        :return: list of index
        """
        return [i for i, j in enumerate(label_list) if j == target]

    def get_score(self, label_list):

        score = 0
        cluster_score = np.zeros(self.C)
        for j in range(self.C):
            index_ = self.get_index(j, label_list)
            nj = len(index_)
            coef = 1 / (nj - 1)

            points_ = self.X[np.array(index_)]
            p_dist = np.log(pdist(points_, 'euclidean'))
            dist_sum = np.sum(p_dist)
            cluster_score[j] = coef * dist_sum
            score += coef * dist_sum
        return cluster_score, score

    def update_score(self, curr_idx, label_list):
        curr_label = label_list[curr_idx]
        new_label_list = [w for w in range(self.C) if w != curr_label]
        temp_label = label_list.copy()
        result_score = {}
        result_c_score = {}
        for new_label in new_label_list:
            temp_label[curr_idx] = new_label
            temp_c_score, temp_score = self.get_score(temp_label)
            result_score[temp_score] = new_label
            result_c_score[temp_score] = temp_c_score
        return min(result_score), result_score[min(result_score)], result_c_score[min(result_score)]

    def exe(self, label_list):

        converged, itr = False, 0
        temp_updated_score, new_cluster_score = 0, np.zeros(self.C)
        while not converged and itr <= self.itr:
            C_score, S_Nic = self.get_score(label_list)
            temp_label_list = []

            updated = 0

            for i in range(len(self.X)):
                temp_updated_score, new_label, new_cluster_score = self.update_score(i, label_list)
                if temp_updated_score < S_Nic:
                    # temp_label_list.append(new_label)
                    # updated += 1
                    S_Nic = temp_updated_score
                    label_list[i] = new_label
                    updated += 1
                # else:
                #     temp_label_list.append(label_list[i])
            # if updated > 0:
            #     label_list = temp_label_list
            # else:
            #     converged = True
            #     break
            if updated > 0:
                converged = True
                break
            itr += 1

        return (temp_updated_score, label_list, new_cluster_score)

    def fit(self, X, n = 10, itr = None):
        # cov = np.cov(X.T)
        # self.X = np.dot(X, cov)
        self.X = whiten(X)
        if itr:
            self.itr = itr
        if n < 1:
            print('wrong iteration number, itreation should greater or equal than 1')
        elif n == 1:
            label_list_ = np.random.randint(0, self.C, len(self.X))
            self.over_all_score, self.labels, self.scores = self.exe(label_list_)
        else:
            label_list_ = [np.random.randint(0, self.C, len(self.X)) for _ in range(n)]

            p = Pool()
            result = p.map(self.exe, label_list_)
            p.close()
            p.join()

            # result_label = self.exe(label_list_)
            temp = result[0][0]
            index = 1
            for i in range(1,n):
                if result[i][0] < temp:
                    index = i
            self.over_all_score,self.labels,self.scores = result[index]


        # self.labels = result_label

    def labels_(self):
        return self.labels

    def get_score_(self):
        return self.over_all_score

def gr(n, w, r, mean=[0, 0], cov=[[1, 0], [0, 1]]):
    x, y = np.random.multivariate_normal(mean, cov, n).T
    xy = zip(x, y)
    result = []
    for z in xy:
        z = np.array(z)
        result.append(z / w + r * z / np.linalg.norm(z))

    return result


if __name__ == '__main__':
    # mean = [0,0]
    # cov = [[1,0],[0,1]]
    # X1 = np.random.multivariate_normal(mean, cov, 500)
    # X2 = np.random.multivariate_normal(mean, cov, 300) + 3
    ring = gr(500, 5, 30)
    x1, y1 = zip(*ring)

    ring2 = gr(500, 5, 5)
    x2, y2 = zip(*ring2)

    ring3 = gr(500,5,15)
    x3,y3 = zip(*ring3)

    ring_m1 = np.array([x1, y1]).T
    ring_m2 = np.array([x2, y2]).T
    ring_m3 = np.array([x3, y3]).T

    X = np.concatenate((ring_m1, ring_m2,ring_m3), axis=0)
    df = pd.DataFrame(X, columns=['X', 'Y'])
    t = NIC(3)
    t.fit(X,5)
    df['label'] = t.labels_()
    sns.scatterplot(df['X'], df['Y'], hue=df['label'])
    plt.show()


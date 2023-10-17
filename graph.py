# ---- coding: utf-8 ----
# @author: Ziyang Zhang
# @description: Load graph from dataset


from collections import defaultdict
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as sparse_inv


class Graph(object):
    def __init__(self, base_dir, feature_file, edge_file, cluster_file, alpha, lambda_):
        self.init(base_dir, feature_file, edge_file, cluster_file, alpha, lambda_)

    def init(self, base_dir, feature_file, edge_file, cluster_file, alpha, lambda_):
        feature = []
        with open(feature_file, 'r') as f:
            for line in f:
                feature.append(np.array(list(map(int, line.rstrip().split(',')))))
        self.feature = np.array(feature)

        cluster = []
        with open(cluster_file, 'r') as f:
            for line in f:
                cluster.append(np.array(list(map(int, line.rstrip().split(',')))))
        self.cluster = np.array(cluster)

        edges = defaultdict(set)
        with open(edge_file, 'r') as f:
            for line in f:
                tuple = list(map(int, line.rstrip().split(',')))
                assert len(tuple) == 2
                edges[tuple[0]].add(tuple[1])
                edges[tuple[1]].add(tuple[0])
        for v, ns in edges.items():
            if v not in ns:
                edges[v].add(v)

        indices = []
        T_values, RI_values, RW_values = [], [], []

        for v, ns in edges.items():
            indices.append(np.array([v, v]))
            T_values.append(1.0 / len(ns))
            RI_values.append(1.0 - alpha / len(ns))
            RW_values.append(1.0 - (1.0 - lambda_) / len(ns))
            for n in ns:
                if v != n:
                    indices.append(np.array([v, n]))
                    T_values.append(1.0 / len(ns))
                    RI_values.append(-alpha / len(ns))
                    RW_values.append(-(1.0 - lambda_) / len(ns))

        self.indices = np.array(indices)
        self.T_values = np.array(T_values, dtype=np.float32)

        sparse_matrix = csc_matrix((RI_values, (self.indices[:, 1], self.indices[:, 0])),
                                   shape=(len(edges), len(edges)))
        self.RI = sparse_inv(sparse_matrix).todense()
        # np.save(base_dir + 'RI.npy', self.RI.astype(np.float32))

        sparse_matrix = csc_matrix((RW_values, (self.indices[:, 0], self.indices[:, 1])),
                                   shape=(len(edges), len(edges)))
        self.RW = lambda_ * sparse_inv(sparse_matrix).todense()
        self.RW /= np.sum(self.RW, axis=0)
        # np.save(base_dir + 'RW.npy', self.RW.astype(np.float32))

        print("######## LOAD GRAPH ########")
        print("features:", self.feature.shape)
        print("clusters:", self.cluster.shape)
        print("indices:", self.indices.shape)
        print("T_values:", self.T_values.shape)
        print("RI:", self.RI.shape)
        print("RW:", self.RW.shape)
        print("#############################")


if __name__ == "__main__":
    # Define parameters
    base_dir = 'data/cora/'
    feature_file = base_dir + 'feature.txt'
    edge_file = base_dir + 'edge.txt'
    cluster_file = base_dir + 'cluster.txt'
    alpha = 0.2  # Example value
    lambda_ = 0.5  # Example value

    # Create a Graph object
    graph = Graph(base_dir, feature_file, edge_file, cluster_file, alpha, lambda_)

    print("features:", graph.feature.shape)
    print("clusters:", graph.cluster.shape)
    print("indices:", graph.indices.shape)
    print("T_values:", graph.T_values.shape)
    print("RI:", graph.RI.shape)
    print("RW:", graph.RW.shape)




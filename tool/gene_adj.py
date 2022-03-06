#coding:utf-8
import numpy as np
from knn import fast_knns2spmat, knns2ordered_nbrs
from adjacency import build_symmetric_adj, row_normalize
from scipy.sparse import coo_matrix, save_npz
import sys

th_sim = 0.0
if __name__ == "__main__":
    knnfile, topk, outfile = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    knn_arr = np.load(knnfile)['data'][:, :, :topk]
    
    adj = fast_knns2spmat(knn_arr, topk, th_sim, use_sim=True)

    # build symmetric adjacency matrix
    adj = build_symmetric_adj(adj, self_loop=True)
    adj = row_normalize(adj)

    adj_coo = adj.tocoo()
    print("edge num", adj_coo.row.shape)
    print("mat shape", adj_coo.shape)
    
    save_npz(outfile, adj)

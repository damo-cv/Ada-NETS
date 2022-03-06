#coding:utf-8
import sys
import numpy as np
from multiprocessing import Pool
import pandas as pd

total_k = 80
def get_topK(query_nodeid):
    query_label = label_arr[query_nodeid]

    total_num = len(np.where(label_arr == query_label)[0])
    prec_list, recall_list, fscore_list = [], [], []
    for topK in range(1, total_k + 1):
        result_list = []
        for i in range(0, topK):
            doc_nodeid = I[query_nodeid][i]
            doc_label = label_arr[doc_nodeid]
            result = 1 if doc_label == query_label else 0
            if i == 0:
                result = 1
            result_list.append(result)
        prec = np.mean(result_list)
        recall = np.sum(result_list) / total_num
        fscore = (1 + beta*beta) * prec * recall / (beta*beta*prec + recall)
        prec_list.append(prec)
        recall_list.append(recall)
        fscore_list.append(fscore)
    fscore_arr = np.array(fscore_list)
    idx = fscore_arr.argmax()
    thres_topK = idx + 1
    return thres_topK, prec_list[idx], recall_list[idx], fscore_list[idx]

if __name__ == "__main__":
    Ifile, labelfile, beta, outfile = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    I = np.load(Ifile)
    label_arr = np.load(labelfile)
    beta = float(beta)

    debug = True
    debug = False
    if debug:
        res = []
        for query_nodeid in range(len(I)):
            item = get_topK(query_nodeid)
            res.append(item)
    else:
        pool = Pool(48)
        res = pool.map(get_topK, range(len(I)))
        pool.close()
        pool.join()

        topK_list, prec_list, recall_list, fscore_list = list(zip(*res))
        np.save(outfile, topK_list)


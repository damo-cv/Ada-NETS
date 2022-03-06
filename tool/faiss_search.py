#coding:utf-8
import numpy as np
import faiss
from tqdm import tqdm
import sys
import time
import os

def batch_search(index, query, topk, bs, verbose=False):
    n = len(query)
    dists = np.zeros((n, topk), dtype=np.float32)
    nbrs = np.zeros((n, topk), dtype=np.int32)

    for sid in tqdm(range(0, n, bs), desc="faiss searching...", disable=not verbose):
        eid = min(n, sid + bs)
        dists[sid:eid], nbrs[sid:eid] = index.search(query[sid:eid], topk)
    cos_dist = dists / 2
    return cos_dist, nbrs


def search(query_arr, doc_arr, outpath, tag, save_file=True):
    ### parameter
    nlist = 100  # 1000 cluster for 100w
    nprobe = 100    # test 10 cluster
    topk = 1024
    bs = 100
    ### end parameter


    #print("configure faiss")
    beg_time = time.time()
    num_gpu = faiss.get_num_gpus()
    dim = query_arr.shape[1]
    #cpu_index = faiss.index_factory(dim, 'IVF100', faiss.METRIC_INNER_PRODUCT)
    quantizer = faiss.IndexFlatL2(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    cpu_index.nprobe = nprobe

    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.usePrecomputed = False
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=num_gpu)

    # start IVF
    #print("build index")
    gpu_index.train(doc_arr)
    gpu_index.add(doc_arr)
    #print(gpu_index.ntotal)

    # start query
    #print("start query")
    gpu_index.nprobe = nprobe # default nprobe is 1, try a few more
    print("beg search")
    D, I = batch_search(gpu_index, query_arr, topk, bs, verbose=True)
    print("time use %.4f"%(time.time()-beg_time))

    if save_file:
        np.save(os.path.join(outpath, tag+'D'), D)
        np.save(os.path.join(outpath, tag+'I'), I)
        data = np.concatenate((I[:,None,:], D[:,None,:]), axis=1)
        np.savez(os.path.join(outpath,'data'), data=data)
    print("time use", time.time()-beg_time)

if __name__ == "__main__":
    queryfile, docfile, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
    if len(sys.argv) == 5:
        tag = sys.argv[4]
    else:
        tag = ""

    query_arr = np.load(queryfile)
    doc_arr = np.load(docfile)
    query_arr = query_arr / np.linalg.norm(query_arr, axis=1, keepdims=True)
    doc_arr = doc_arr / np.linalg.norm(doc_arr, axis=1, keepdims=True)

    search(query_arr, doc_arr, outpath, tag)

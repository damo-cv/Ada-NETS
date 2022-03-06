#coding:utf-8
import sys
import numpy as np
from multiprocessing import Pool, Manager
import copy
from functools import partial

k =  80
lamb = 0.3

def worker(k, queue, query_nodeid):
    docnodeid_list = I[query_nodeid, :k]
    query_Rstarset = Rstarset_list[query_nodeid]
    outlist = []
    for idx, doc_nodeid in enumerate(docnodeid_list):
        doc_Rstarset = Rstarset_list[doc_nodeid]
        sim = 1.0 * len(query_Rstarset & doc_Rstarset) / len(query_Rstarset | doc_Rstarset)
        jd = 1 - sim
        cd = D[query_nodeid, idx]
        nd = (1-lamb) * jd + lamb * cd
        tpl = (doc_nodeid, nd)
        outlist.append(tpl)
    outlist = sorted(outlist, key=lambda x:x[1])
    queue.put(query_nodeid)
    fn_name = sys._getframe().f_code.co_name
    #if queue.qsize() % 1000 == 0:
    #    print("==>", fn_name, queue.qsize())
    return list(zip(*outlist))

def get_Kngbr(query_nodeid, k):
    Kngbr = I[query_nodeid, :k]
    return set(Kngbr)

def get_Rset(k, queue, query_nodeid):
    docnodeid_set = get_Kngbr(query_nodeid, k)
    Rset = set()
    for doc_nodeid in docnodeid_set:
        if query_nodeid not in get_Kngbr(doc_nodeid, k):
            continue
        Rset.add(doc_nodeid)
    queue.put(query_nodeid)
    fn_name = sys._getframe().f_code.co_name
    #if queue.qsize() % 1000 == 0:
    #    print("==>", fn_name, queue.qsize())
    return Rset

def get_Rstarset(queue, query_nodeid):
    Rset = Rset_list[query_nodeid]
    Rstarset = copy.deepcopy(Rset)
    for doc_nodeid in Rset:
        doc_Rset = half_Rset_list[doc_nodeid]
        if len(doc_Rset & Rset) >= len(doc_Rset) * 2 / 3:
            Rstarset |= doc_Rset
    queue.put(query_nodeid)
    fn_name = sys._getframe().f_code.co_name
    #if queue.qsize() % 1000 == 0:
    #    print("==>", fn_name, queue.qsize())
    return Rstarset

if __name__ == "__main__":
    Ifile, Dfile, topk, outIfile, outDfile, outDatafile = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
    k = int(topk)
    print("use topk", k)
    I = np.load(Ifile)
    D = np.load(Dfile)

    queue1 = Manager().Queue()
    queue2 = Manager().Queue()
    queue3 = Manager().Queue()
    queue4 = Manager().Queue()
    debug = True
    debug = False
    if debug:
        for query_nodeid in range(len(I)):
            res = worker(k, query_nodeid)
    else:
        pool = Pool(52)
        get_Rset_partial = partial(get_Rset, k, queue1)
        Rset_list = pool.map(get_Rset_partial, range(len(I)))
        pool.close()
        pool.join()

        pool = Pool(52)
        k2 = k // 2
        get_Rset_partial = partial(get_Rset, k2, queue2)
        half_Rset_list = pool.map(get_Rset_partial, range(len(I)))
        pool.close()
        pool.join()

        pool = Pool(52)
        get_Rstarset_partial = partial(get_Rstarset, queue3)
        Rstarset_list = pool.map(get_Rstarset_partial, range(len(I)))
        pool.close()
        pool.join()

        pool = Pool(52)
        worker_partial = partial(worker, k, queue4)
        res = pool.map(worker_partial, range(len(I)))
        pool.close()
        pool.join()

        newI, newD = list(zip(*res))
        newI = np.array(newI)
        newD = np.array(newD)
        newdata = np.concatenate((newI[:,None,:], newD[:,None,:]), axis=1)
        np.save(outIfile, newI)
        np.save(outDfile, newD)
        np.savez(outDatafile, data=newdata)

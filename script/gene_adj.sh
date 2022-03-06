#!/bin/bash
set -euxo pipefail
set +x

mkdir -p data/adj/train data/adj/test

knnfile=data/knn/train/data.npz
topk=256
outfile=data/adj/train/adj
#python tool/gene_adj.py $knnfile $topk $outfile

knnfile=data/ss/test/data.npz
kfile=AND/outpath/k_infer_pred.npy
outfile=data/adj/test/adj_adanets
python tool/gene_adj_adanets.py $knnfile $kfile $outfile

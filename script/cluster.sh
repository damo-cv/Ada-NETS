#!/bin/bash
set -euxo pipefail
set +x

cd GCN

featfile=outpath/fcfeat_ckpt_35000.npy
tag=fc
python -W ignore ../tool/faiss_search.py $featfile $featfile $outpath $tag

Ifile=outpath/fcI.npy
Dfile=outpath/fcD.npy
python cluster.py $Ifile $Dfile $test_labelfile

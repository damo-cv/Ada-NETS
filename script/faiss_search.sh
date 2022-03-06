#!/bin/bash
set -euxo pipefail
set +x

mkdir -p data/knn/train data/knn/test
featfile=data/feature/part0_train.npy
outpath=data/knn/train
python -W ignore tool/faiss_search.py $featfile $featfile $outpath

featfile=data/feature/part1_test.npy
outpath=data/knn/test
python -W ignore tool/faiss_search.py $featfile $featfile $outpath

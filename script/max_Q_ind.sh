#!/bin/bash
set -euxo pipefail
set +x

mkdir -p data/max_Q/train data/max_Q/test

beta=0.50
Ifile=data/ss/train/I.npy
labelfile=data/label/part0_train.npy
outfile=data/max_Q/train/ind
python tool/max_Q_ind.py $Ifile $labelfile $beta $outfile

Ifile=data/ss/test/I.npy
labelfile=data/label/part1_test.npy
outfile=data/max_Q/test/ind
python tool/max_Q_ind.py $Ifile $labelfile $beta $outfile

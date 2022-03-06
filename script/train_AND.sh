#!/bin/bash
set -euxo pipefail
set +x

cd AND
outpath=outpath
mkdir -p $outpath

train_featfile=../data/feature/part0_train.npy
train_Ifile=../data/ss/train/I.npy
train_labelfile=../data/max_Q/train/ind.npy

test_featfile=../data/feature/part1_test.npy
test_Ifile=../data/ss/test/I.npy
test_labelfile=../data/max_Q/test/ind.npy

phase=train
param=" --config_file config.yml --outpath $outpath --phase $phase
        --train_featfile $train_featfile --train_Ifile $train_Ifile --train_labelfile $train_labelfile
        --test_featfile $test_featfile --test_Ifile $test_Ifile --test_labelfile $test_labelfile"
#python -u train.py $param

phase=test
ckpt=ckpt_40000.pth
param=" --config_file config.yml --outpath $outpath --phase $phase
        --train_featfile $train_featfile --train_Ifile $train_Ifile --train_labelfile $train_labelfile
        --test_featfile $test_featfile --test_Ifile $test_Ifile --test_labelfile $test_labelfile
        --resume_path ${outpath}/ckpt_40000.pth"
python -u train.py ${param}

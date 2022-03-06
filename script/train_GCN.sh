#!/bin/bash
set -euxo pipefail
set +x

cd GCN

losstype=allmax
margin=1.0
pweight=1.0
pmargin=0.9
beta=0.50

outpath=outpath
train_featfile=../data/feature/part0_train.npy
train_orderadjfile=../data/adj/train/adj.npz
train_adjfile=../data/adj/train/adj.npz
train_labelfile=../data/label/part0_train.npy
test_featfile=../data/feature/part1_test.npy
test_adjfile=../data/adj/test/adj_adanets.npz
test_labelfile=../data/label/part1_test.npy

phase=train
param="--config_file config.yml --outpath $outpath --phase $phase
    --train_featfile $train_featfile --train_adjfile $train_adjfile --train_labelfile $train_labelfile --train_orderadjfile $train_orderadjfile
    --test_featfile $test_featfile --test_adjfile $test_adjfile --test_labelfile $test_labelfile
    --losstype $losstype --margin $margin --pweight $pweight --pmargin ${pmargin}"
python -u train.py $param

phase=test
param="--config_file config.yml --outpath $outpath --phase $phase
    --train_featfile $train_featfile --train_adjfile $train_adjfile --train_labelfile $train_labelfile --train_orderadjfile $train_orderadjfile
    --test_featfile $test_featfile --test_adjfile $test_adjfile --test_labelfile $test_labelfile
    --losstype $losstype --margin $margin --pweight $pweight --pmargin ${pmargin}  --resume_path $outpath/ckpt_35000.pth"
python -u train.py $param

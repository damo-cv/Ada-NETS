#!/bin/bash
set -euxo pipefail
set +x

mkdir -p data/ss/train data/ss/test
python tool/struct_space.py  data/knn/test/I.npy  data/knn/test/D.npy 80  data/ss/test/I  data/ss/test/D  data/ss/test/data
python tool/struct_space.py data/knn/train/I.npy data/knn/train/D.npy 80 data/ss/train/I data/ss/train/D data/ss/train/data

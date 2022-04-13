# Ada-NETS

This is an official implementation for "Ada-NETS: Face Clustering via Adaptive Neighbour Discovery in the Structure Space" accepted at ICLR 2022.



## Introduction

This paper presents a novel Ada-NETS algorithm to deal with the noise edges problem when building the graph in GCN-based face clustering. In Ada-NETS, the features are first transformed to the structure space to enhance the accuracy of the similarity metrics. Then an adaptive neighbour discovery method is used to find neighbours for all samples adaptively with the guidance of a heuristic quality criterion. Based on the discovered neighbour relations, a graph with clean and rich edges is built as the input of GCNs to obtain state-of-the-art on the face, clothes, and person clustering tasks.

<img src=image/fig.png width=1000 height=345 />



## Main Results

<img src=image/results.png width=900 height=355 />




## Getting Started

### Install

+ Clone this repo

```
git clone https://github.com/Thomas-wyh/Ada-NETS
cd Ada-NETS
```

+ Create a conda virtual environment and activate it

```
conda create -n adanets python=3.6 -y
conda activate adanets
```

+ Install `Pytorch` , `cudatoolkit` and other requirements.
```
conda install pytorch==1.2 torchvision==0.4.0a0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

- Install `Apex`:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Data preparation

The process of clustering on the MS-Celeb part1 is as follows:

The original data files are from [here](https://github.com/yl-1993/learn-to-cluster/blob/master/DATASET.md#supported-datasets)(The feature and label files of MSMT17 used in Ada-NETS are [here](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/Ada-NETS/MSMT17/msmt17_feature_label.zip)). For convenience, we convert them to `.npy` format after L2 normalized. The original features' dimension is 256. The file structure should look like:

```
data
├── feature
│   ├── part0_train.npy
│   └── part1_test.npy
└── label
    ├── part0_train.npy
    └── part1_test.npy
```

Build the $k$NN by faiss:

```
sh script/faiss_search.sh
```

Obtain the top$K$ neighbours and distances of each vertex in the structure space:

```
sh script/struct_space.sh
```

Obtain the best neigbours by the candidate neighbours quality criterion:

```
sh script/max_Q_ind.sh
```

### Train the Adaptive Filter

Train the adaptive filter based on the data prepared above:

```
sh script/train_AND.sh
```

### Train the GCN and cluster faces

Generate the clean yet rich Graph:

```
sh script/gene_adj.sh
```

Train the GCN to obtain enhanced vertex features:

```
sh script/train_GCN.sh
```

Perform cluster faces:

```
sh script/cluster.sh
```

It will print the evaluation results of clustering. The Bcubed F-socre is about 91.4 and the Pairwise F-score is about 92.7.



## Acknowledgement

This code is based on the publicly available face clustering [codebase](https://github.com/yl-1993/learn-to-cluster), [codebase](https://github.com/makarandtapaswi/BallClustering_ICCV2019) and the [dmlc/dgl](https://github.com/dmlc/dgl).

The k-nearest neighbor search tool uses [faiss](https://github.com/facebookresearch/faiss).




## Citing Ada-NETS

```
@inproceedings{wang2022adanets,
  title={Ada-NETS: Face Clustering via Adaptive Neighbour Discovery in the Structure Space},
  author={Yaohua Wang and Yaobin Zhang and Fangyi Zhang and Senzhang Wang and Ming Lin and YuQi Zhang and Xiuyu Sun},
  booktitle={International conference on learning representations (ICLR)},
  year={2022}
}

@misc{wang2022adanets,
      title={Ada-NETS: Face Clustering via Adaptive Neighbour Discovery in the Structure Space}, 
      author={Yaohua Wang and Yaobin Zhang and Fangyi Zhang and Senzhang Wang and Ming Lin and YuQi Zhang and Xiuyu Sun},
      year={2022},
      eprint={2202.03800},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

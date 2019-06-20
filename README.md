ClusterGCN
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixhop-higher-order-graph-convolution/node-classification-on-citeseer)](https://paperswithcode.com/sota/node-classification-on-citeseer?p=mixhop-higher-order-graph-convolution)
<img src="https://img.shields.io/badge/license-MIT-blue.svg"/>
============================================
A PyTorch implementation of "Cluster-GCN: An Efficient Algorithm for Training Deep andLarge Graph Convolutional Networks" (KDD 2019).
<p align="center">
  <img width="600" src="clustergcn.jpg">
</p>
### Abstract
<p align="justify">
Graph convolutional network (GCN) has been successfully applied to many graph-based applications; however, training a large-scale GCN remains challenging. Current SGD-based algorithms suffer from either a high computational cost that exponentially grows with number of GCN layers, or a large space requirement for keeping the entire graph and the embedding of each node in memory. In this paper, we propose Cluster-GCN, a novel GCN algorithm that is suitable for SGD-based training by exploiting the graph clustering structure. Cluster-GCN works as the following: at each step, it samples a block of nodes that associate with a dense subgraph identified by a graph clustering algorithm, and restricts the neighborhood search within this subgraph. This simple but effective strategy leads to significantly improved memory and computational efficiency while being able to achieve comparable test accuracy with previous algorithms. To test the scalability of our algorithm, we create a new Amazon2M data with 2 million nodes and 61 million edges which is more than 5 times larger than the previous largest publicly available dataset (Reddit). For training a 3-layer GCN on this data, Cluster-GCN is faster than the previous state-of-the-art VR-GCN (1523 seconds vs 1961 seconds) and using much less memory (2.2GB vs 11.2GB). Furthermore, for training 4 layer GCN on this data, our algorithm can finish in around 36 minutes while all the existing GCN training algorithms fail to train due to the out-of-memory issue. Furthermore, Cluster-GCN allows us to train much deeper GCN without much time and memory overhead, which leads to improved prediction accuracy---using a 5-layer Cluster-GCN, we achieve state-of-the-art test F1 score 99.36 on the PPI dataset, while the previous best result was 98.71 b</p>

This repository provides a PyTorch implementation of MixHop and N-GCN as described in the papers:

> MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing
> Sami Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Hrayr Harutyunyan, Nazanin Alipourfard, Kristina Lerman, Greg Ver Steeg, and Aram Galstyan.
> ICML, 2019.
> [[Paper]](https://arxiv.org/pdf/1905.00067.pdf)

> A Higher-Order Graph Convolutional Layer.
> Sami A Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Hrayr Harutyunyan.
> NeurIPS, 2018.
> [[Paper]](http://sami.haija.org/papers/high-order-gc-layer.pdf)

The original TensorFlow implementation is available [[here]](https://github.com/samihaija/mixhop).

### Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          1.11
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
torch             0.4.1
torch-sparse      0.2.2
```
### Datasets

The code takes the **edge list** of the graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. A sample graph for `Cora` is included in the  `input/` directory. In addition to the edgelist there is a JSON file with the sparse features and a csv with the target variable.

The **feature matrix** is a sparse binary one it is stored as a json. Nodes are keys of the json and feature indices are the values. For each node feature column ids are stored as elements of a list. The feature matrix is structured as:

```javascript
{ 0: [0, 1, 38, 1968, 2000, 52727],
  1: [10000, 20, 3],
  2: [],
  ...
  n: [2018, 10000]}
```
The **target vector** is a csv with two columns and headers, the first contains the node identifiers the second the targets. This csv is sorted by node identifiers and the target column contains the class meberships indexed from zero. 

| **NODE ID**| **Target** |
| --- | --- |
| 0 | 3 |
| 1 | 1 |
| 2 | 0 |
| 3 | 1 |
| ... | ... |
| n | 3 |

### Options
Training an N-GCN/MixHop model is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options
```
  --edge-path       STR    Edge list csv.         Default is `input/cora_edges.csv`.
  --features-path   STR    Features json.         Default is `input/cora_features.json`.
  --target-path     STR    Target classes csv.    Default is `input/cora_target.csv`.
```
#### Model options
```
  --model             STR     Model variant.                 Default is `mixhop`.               
  --seed              INT     Random seed.                   Default is 42.
  --epochs            INT     Number of training epochs.     Default is 2000.
  --early-stopping    INT     Early stopping rounds.         Default is 10.
  --training-size     INT     Training set size.             Default is 1500.
  --validation-size   INT     Validation set size.           Default is 500.
  --learning-rate     FLOAT   Adam learning rate.            Default is 0.01.
  --dropout           FLOAT   Dropout rate value.            Default is 0.5.
  --lambd             FLOAT   Regularization coefficient.    Default is 0.0005.
  --layers-1          LST     Layer sizes (upstream).        Default is [200, 200, 200]. 
  --layers-2          LST     Layer sizes (bottom).          Default is [200, 200, 200].
  --cut-off           FLOAT   Norm cut-off for pruning.      Default is 0.1.
  --budget            INT     Architecture neuron budget.    Default is 60.
```
### Examples
The following commands learn a neural network and score on the test set. Training a model on the default dataset.
```
python src/main.py
```
<p align="center">
<img style="float: center;" src="mixhop.gif">
</p>

Training a MixHop model for a 100 epochs.
```
python src/main.py --epochs 100
```
Increasing the learning rate and the dropout.
```
python src/main.py --learning-rate 0.1 --dropout 0.9
```
Training a model with diffusion order 2:
```
python src/main.py --layers 64 64
```
Training an N-GCN model:
```
python src/main.py --model ngcn
```

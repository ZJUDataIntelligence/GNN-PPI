# GNN-PPI

## 1. Overview

This repository is the implementation of "Graph Neural Network for Protein-Protein Interaction Prediction: A Comparative Study". It is based on the [hgcn](https://github.com/HazyResearch/hgcn) repository with additional PPI data and some code optimization. All models including NN, GCN, GAT, HNN, HGCN are implementated with Pytorch.


## 2. Setup

### 2.1 Installation with conda

If you don't have conda installed, please install it following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```git clone https://github.com/ZJUDataIntelligence/GNN-PPI.git```

```cd GNN-PPI```

```conda env create -f environment.yml```

### 2.2 Installation with pip

Alternatively, if you prefer to install dependencies with pip, please follow the instructions below:

```virtualenv -p [PATH to python3.7 binary] hgcn```

```source hgcn/bin/activate```

```pip install -r requirements.txt```

## 3. Usage

### 3.1 ```set_env.sh```

Before training, run 

```source set_env.sh```

This will create environment variables that are used in the code. 

### 3.2  ```train.py```

This script trains models for link prediction and node classification tasks. 
Metrics are printed at the end of training or can be saved in a directory by adding the command line argument ```--save=1```.

```
optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --dropout DROPOUT     dropout probability
  --cuda CUDA           which cuda device to use (-1 for cpu training)
  --epochs EPOCHS       maximum number of epochs to train for
  --weight-decay WEIGHT_DECAY
                        l2 regularization strength
  --optimizer OPTIMIZER
                        which optimizer to use, can be any of [Adam,
                        RiemannianAdam]
  --momentum MOMENTUM   momentum in optimizer
  --patience PATIENCE   patience for early stopping
  --seed SEED           seed for training
  --log-freq LOG_FREQ   how often to compute print train/val metrics (in
                        epochs)
  --eval-freq EVAL_FREQ
                        how often to compute val metrics (in epochs)
  --save SAVE           1 to save model and logs and 0 otherwise
  --save-dir SAVE_DIR   path to save training logs and model weights (defaults
                        to logs/task/date/run/)
  --sweep-c SWEEP_C
  --lr-reduce-freq LR_REDUCE_FREQ
                        reduce lr every lr-reduce-freq or None to keep lr
                        constant
  --gamma GAMMA         gamma for lr scheduler
  --print-epoch PRINT_EPOCH
  --grad-clip GRAD_CLIP
                        max norm for gradient clipping, or None for no
                        gradient clipping
  --min-epochs MIN_EPOCHS
                        do not early stop before min-epochs
  --task TASK           which tasks to train on, can be any of [lp, nc]
  --model MODEL         which encoder to use, can be any of [Shallow, MLP,
                        HNN, GCN, GAT, HGCN]
  --dim DIM             embedding dimension
  --manifold MANIFOLD   which manifold to use, can be any of [Euclidean,
                        Hyperboloid, PoincareBall]
  --c C                 hyperbolic radius, set to None for trainable curvature
  --r R                 fermi-dirac decoder parameter for lp
  --t T                 fermi-dirac decoder parameter for lp
  --pretrained-embeddings PRETRAINED_EMBEDDINGS
                        path to pretrained embeddings (.npy file) for Shallow
                        node classification
  --pos-weight POS_WEIGHT
                        whether to upweight positive class in node
                        classification tasks
  --num-layers NUM_LAYERS
                        number of hidden layers in encoder
  --bias BIAS           whether to use bias (1) or not (0)
  --act ACT             which activation function to use (or None for no
                        activation)
  --n-heads N_HEADS     number of attention heads for graph attention
                        networks, must be a divisor dim
  --alpha ALPHA         alpha for leakyrelu in graph attention networks
  --use-att USE_ATT     whether to use hyperbolic attention in HGCN model
  --double-precision DOUBLE_PRECISION
                        whether to use double precision
  --dataset DATASET     which dataset to use
  --val-prop VAL_PROP   proportion of validation edges for link prediction
  --test-prop TEST_PROP
                        proportion of test edges for link prediction
  --use-feats USE_FEATS
                        whether to use node features or not
  --normalize-feats NORMALIZE_FEATS
                        whether to normalize input node features
  --normalize-adj NORMALIZE_ADJ
                        whether to row-normalize the adjacency matrix
  --split-seed SPLIT_SEED
                        seed for data splits (train/test/val)
  --ppitype DATASET
                        using SHS27 or SHS248 dataset
  --ppimode DATASET
                        ppi mode, including activation, binding, catalysis, expression, inhibition, ptmod or reaction
```

## 4. Examples

#### On SHS27-activation dataset using HGCN


```python train.py --task lp --dataset ppi --model HGCN --lr 0.01 --dim 16 --num-layers 2 --act relu --bias 1 --dropout 0.4 --weight-decay 0.0001 --manifold PoincareBall --cuda 0 --log-freq 5 --ppitype SHS27 --ppimode activation```



# Causal Inference Based Debiasing Framework for Knowledge Graph Completion

Code for Causal Inference Based Debiasing Framework for Knowledge Graph Completion



In this paper, we conduct a comprehensive analysis of these biases to determine their extent of impact. To mitigate these biases, we propose a debiasing framework called Causal Inference-based Debiasing Framework for KGC (CIDF) by formulating a causal graph and utilizing it for causal analysis of KGC tasks.



In this project, we employ CIDF on  [SimKGC](https://github.com/intfloat/simkgc).

# Requirements

```
python=3.7
torch=1.11.0+cu113
transformers=4.27.1
wandb
```

All experiments are run with 4 V100(32GB) GPUs.

# Download Datasets

For datasets WN18RR and FB25k-127, [SimKGC](https://github.com/intfloat/simkgc) provides their resources for downloading.

For datasets [Wikidata5M](https://deepgraphlearning.github.io/project/wikidata5m), it can be downloaded as following:

```
bash ./scripts/download_wikidata5m.sh
```

The directory structure is as follows:

```
data
|- FB15k237
|- WN18RR
|- wikidata5m
```

# Preprocess Datasets

After downloading the datasets, use the following bash commands to preprocess them.

```
bash scripts/preprocess.sh WN18RR

bash scripts/preprocess.sh FB15k237

bash scripts/preprocess.sh wiki5m_trans

bash scripts/preprocess.sh wiki5m_ind
```

# Train and Evaluate

You can use the following bash commands to train and evaluate CIDF

```
OUTPUT_DIR=<model_saved_path> bash <script_path> <logger_name_of_wandb>

OUTPUT_DIR=./checkpoint/wn18rr/ bash scripts/train_wn.sh wn

OUTPUT_DIR=./checkpoint/fb15k237/ bash scripts/train_fb.sh fb

OUTPUT_DIR=./checkpoint/wikiT/ bash scripts/train_wiki_trans.sh wiki5m_trans

OUTPUT_DIR=./checkpoint/wikiI/ bash scripts/train_wiki_ind.sh wiki5m_ind
```

# Evaluate Only

**train_args.bin** will be created in **model_saved_path** when training is finished.

You can reload the training set and evaluate the model as follow:

```
python evaluate.py -apath <model_saved_path>/train_args.bin
```

# Citation

Citation will be uploaded soon
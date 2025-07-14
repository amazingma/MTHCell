# MTHCell: Multiscale Multiview Transformers for Cell Type Identification
![MTHCell](https://github.com/amazingma/MTHCell/blob/main/figures/model.png)
This repo provides the source code & data of our paper **MTHCell**: a **M**ultiview **T**ransformer-based **H**ierarchical Fusion Model for **Cell** Type Identification.

## Overview
Identifying cellular identities is one of the first steps in the analysis of single-cell RNA sequencing (scRNA-seq) data. Although a number of methods have been developed for this purpose, such tools suffer from a lack of curated marker gene lists, improper handling of batch effects, and struggle to strike a balance between accuracy and interpretability. To overcome these challenges, we develop MTHCell, which introduces multi-view biological knowledge encoding and multi-scale feature learning to transformers. At the view level, the supervised learning of modal features and the imposing of distance constraints between different views allow the network to achieve a good balance between learning common information and discrepancy information across diverse views. At the instance level, the mechanism for dynamically discovering the 'most similar' class in each epoch/batch allows the network to focus on separating the samples from the most similar non-self-class samples, resulting in a more uniform distribution of the representation space. We apply MTHCell to human and mouse scRNA-seq datasets from various tissues. Extensive and rigorous benchmark studies validate the superior performance of MTHCell in cell type annotation, rare and novel cell type discovery, robustness to batch effects and model interpretability.
## Getting Started
### 1. Clone the repository
```
git clone https://github.com/amazingma/MTHCell.git
```
### 2. Create conda environment
```
conda create -n MTHCell python=3.8
conda activate MTHCell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
or pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
### 3. Environment has been tested
`environment.yaml`
### 4. Obtain Datasets
The prepared benchmarking set is available at https://zenodo.org/records/14728964.

## Usage
### 1. Activate the created conda environment
```
source activate MTHCell
```
### 2. Train the model
```
python ./code/train.py
```
To train on your own dataset, you need to provide training and test data, place them in `./data`, and modify the `train_path` and `test_path` parameters. SGD is chosen as the optimizer, and we use cosine learning rate decay to avoid too large steps in the late stages of training.
### 3. Key parameters
* `max_g`: Maximum number of gene members in a pathway or regulon.<br/>
* `max_gs`: Maximum number of pathways or regulons.<br/>
* `embed_dim`: Embedding dimension.<br/>
* `depth`: Depth of transformer.<br/>
* `num_heads`: Number of attention heads.<br/>
* `batch_size`: Batch size.<br/>
* `lr`: Initial learning rate.<br/>
* `lrf`: Controlling the decay rate of the lr.<br/>
* `label_name`: Column name of cell type in adata.obs.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact
If you have any questions, please feel free to contact the authors.
Teng Ma: mateng@csu.edu.cn

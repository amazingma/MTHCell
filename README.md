# MMTCell: Multiscale Multiview Transformers for Cell Type Annotation
![MMTCell](https://github.com/amazingma/MMTCell/blob/main/figures/model.png)
This repo provides the source code & data of our paper **MMTCell**: **M**ultiscale **M**ultiview **T**ransformers for **Cell** Type Annotation.

## Overview
One of the first steps in the analysis of single-cell RNA sequencing (scRNA-seq) data is the assignment of cell types. Although a number of methods have been developed for this purpose, such annotation tools suffer from a lack of curated marker gene lists, improper handling of batch effects, and struggle to strike a balance between accuracy and interpretability. To overcome these challenges, we develop MMTCell, which introduces multi-view biological knowledge encoding and multi-scale feature learning to transformers. At the view level, the supervised learning of modal features and the imposing of distance constraints between different views allow the network to achieve a good balance between learning common information and discrepancy information across diverse views. At the instance level, the mechanism of dynamically discovering the 'most similar' class in each epoch/batch allows the network to focus on separating the samples from the most similar non-self-class samples, resulting in a more uniform distribution of the representation space. We apply MMTCell to human and mouse scRNA-seq datasets from various tissues. Extensive and rigorous benchmark studies validate the superior performance of MMTCell in cell type annotation, robustness to batch effects, novel cell type discovery and model interpretability.

## Getting Started
### 1. Clone the repo
```
git clone https://github.com/amazingma/MMTCell.git
```
### 2. Create conda environment
```
conda create -n mmtcell python=3.8
conda activate mmtcell
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```
### 3. Environment has been tested
`environment.yaml`

## Usage
### 1. Activate the created conda environment
```
source activate mmtcell
```
### 2. Train the model
```
python ./code/train.py
```
To train on your own dataset, you need to provide training and test data, place them in `./data`, and modify the parameter `train_path` and `test_path`.
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

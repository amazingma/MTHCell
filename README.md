# MMTCell: Multiscale Multiview Transformers for Cell Type Annotation
![MMTCell](https://github.com/amazingma/MMTCell/blob/main/figures/model.pdf)

## Introduction
One of the first steps in the analysis of single-cell RNA sequencing (scRNA-seq) data is the assignment of cell types. Although a number of methods have been developed for this, such annotation tools suffer from a lack of curated marker gene lists, improper handling of batch effects, and struggle to strike a balance between accuracy and interpretability. To overcome these challenges, we develop MMTCell, which introduces multi-view biological knowledge encoding and multi-scale feature learning to transformers. At the view level, the supervised learning of modal features and the imposing of distance constraints between different views allow the network to achieve a good balance between learning common information and discrepancy information across diverse views. At the instance level, the mechanism of dynamically discovering the 'most similar' class in each epoch/batch allows the network to focus on separating the samples from the most similar non-self-class samples, resulting in a more uniform distribution of the representation space. We apply MMTCell to human and mouse scRNA-seq datasets from various tissues. Extensive and rigorous benchmark studies validate the superior performance of MMTCell in cell type annotation, robustness to batch effects, novel cell type discovery and model interpretability.

## Getting Started
### 1. Clone the repo
```
git clone https://github.com/amazingma/MMTCell.git
```
### 2. Create conda environment
```
conda env create --name mmtcell --file=environment.yml
```

## Usage
### 1. Activate the created conda environment
```
source activate mmtcell
```
### 2. Train the model
```
python train.py
```

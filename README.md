# SSLs-Multiplexed-Imaging
Code for ["Self-supervised learning enables unbiased patient characterization from multiplexed microscopy images"](https://www.biorxiv.org/content/10.1101/2025.03.05.640729v1)

- [DINO](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper), [MAE](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper), [SIMCLR](https://arxiv.org/abs/2002.05709), and [VICRegL](https://arxiv.org/abs/2210.01571) methods are used for level 1 training.
- [DINO](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper) and [MAE](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper) methods are used for level 2 training.

Each subdirectory contains one method.

#### Requirements

```python
torch==2.2.1
torchvision==0.17.1
lightly==1.4.26
pytorch-lightning==2.2
timm==0.9.12
h5py==3.7.0
```
and other regular libraries.

#### For training

- Please modify `chunked_h5_dataset.py` files according to your own dataset.
- Prerequisits for the dataset:
    - Crop the images into (256x256px) small patches.
    - The image values should be log transformations if the dataset histogram is highly skewed.
- `main.py` trains a model, and hyperparameters can be changed in this file.
    
#### For extracting embeddings

1. `get_embeddings.py` files extract the feature representations. Feature representations will be L2 normalized.

#### Creating attention maps

- Attention maps of one TMA core can be created with `level_1_code/MAE_Lightly/AttentionMap.ipynb` notebook.
- The notebook inputs one TMA core image, and creates twelve attention maps for that core.
- Trained models of Level-1 MAE method can be found here in https://zenodo.org/records/15011375.

The code is based on Lightly SSL library: https://github.com/lightly-ai/lightly
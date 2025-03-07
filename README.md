# SSLs-Multiplexed-Imaging
 Code for [Self-supervised learning enables unbiased patient characterization from multiplexed microscopy images](https://www.biorxiv.org/content/10.1101/2025.03.05.640729v1)

The code is based on Lightly SSL library: https://github.com/lightly-ai/lightly

- [DINO](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper), [MAE](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper), [SIMCLR](https://arxiv.org/abs/2002.05709), and [VICRegL](https://arxiv.org/abs/2210.01571) methods are used for level 1 training.
- DINO and MAE methods are used for level 2 training.
- Attention maps of one TMA core can be created with `./level_1_code/MAE_Lightly/AttentionMap.ipynb` notebook.


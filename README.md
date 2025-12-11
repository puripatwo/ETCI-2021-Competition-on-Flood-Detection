# High-Resolution Flood Segmentation in Thailand through Sentinel-1 SAR Imagery

## Background

This repository has been forked from [ETCI-2021-Competition-on-Flood-Detection](https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection/tree/main/notebooks), and is based on their models [[1]](#1). Here, the ETCI-2021 Flooding Dataset will be replaced with the UNOSAT Flood Dataset from [UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service](https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service?tab=readme-ov-file). This is due to how the latter has their own curated dataset with expert analysis on a couple of countries including Thailand [[2]](#2). The goal is to set the images from Thailand as the validation and test set, allowing the model to generalise to Thailand through semi-supervised learning.

## Setup

`git clone https://github.com/puripatwo/ETCI-2021-Competition-on-Flood-Detection.git`

`pip install requirements.txt`

NOTE: You can use a Conda environment or a virtual environment here.

## Data Preparation

We will be focusing on turning the UNOSAT Flood Dataset into a usable dataset to replace the ETCI-dataset.

1. Login to [Earthdata] (https://urs.earthdata.nasa.gov/profile)
2. Ensure you go to Applications, then Authorized Apps, and make sure you approve HyP3.
3. Now, go to `hyp3_downloader.ipynb`, and run the cells there. Make sure you enter your username and password.
4. Here, you will have to rearrange your files. You will create a folder called UNOSAT-Dataset. The folder should be as follows:

```
UNOSAT-Dataset/
│
├── S1A_IW_GRDH_1SDV_20180501T025444_20180501T025509_021705_025710_8835/
│   ├── <VV file>.tif
│   └── <VH file>.tif
│
│
├── S1A_IW_GRDH_1SDV_20190906T110524_20190906T110549_028900_0346B3_457F/
│   ├── <VV file>.tif
│   ├── <VH file>.tif
│   └── ...
│
...
```

5. Once the files have all been downloaded, go to `preprocessing.ipynb`. Run the cells there. They will tile each of the satellite images and place them in OUT_DIR.

The final-UNOSAT dataset should have the follow structure:

```
final-UNOSAT-Dataset/
│
├── train/
│   ├── <SceneA>/
│   │   └── tiles/
│   │       ├── vv/
│   │       │   ├── tile_001_vv.png
│   │       │   ├── tile_002_vv.png
│   │       │   └── ...
│   │       ├── vh/
│   │       │   ├── tile_001_vh.png
│   │       │   ├── tile_002_vh.png
│   │       │   └── ...
│   │       └── label/
│   │           ├── tile_001_label.png
│   │           ├── tile_002_label.png
│   │           └── ...
│   │
│   ├── <SceneB>/
│   └── ...
│
├── test/
│   ├── <SceneX>/
│   │   └── tiles/
│   │       ├── vv/
│   │       ├── vh/
│   │       └── label/
│   └── ...
│
└── test_internal/
    ├── <SceneY>/
    │   └── tiles/
    │       ├── vv/
    │       ├── vh/
    │       └── label/
    └── ...
```

6. If you will be running the models in Google Colab, zip the file and upload it to Google Drive.

## Model Pipeline

For the purpose of this section, we will be running the models on Google Colab. Everything in Google Colab should be set up enough for you to run the models.

`run.ipynb` follows the model proposed this [paper](https://arxiv.org/abs/2107.08369). There are 4 main stages:

   1. Train two models: U-Net and U-Net++ with `train.py`.
   2. With `generate_pseudo.py`, we put together an ensemble of the two models (three if not in round 0) and generate pseudo labels.
   3. We then fine-tune the U-Net model using a combined dataset of the training dataset and the pseudo labels with `train_pseudo.py`.
   4. We can repeat steps 1-3 for how many times you like. We create a new U-Net and U-Net++ each time, but keep fine-tuning the U-Net model from the previous iteration.
   5. Build an ensemble of the three models and then get the inference and evaluation from `ensemble_inference.py`.

## References

<a id="1">[1]</a> Paul, S. and Ganju, S. (2021) Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning, arXiv.org. Available at: https://arxiv.org/abs/2107.08369 (Accessed: 11 December 2025).

<a id="2">[2]</a> Nemni, E., Bullock, J., Belabbes, S. and Bromley, L. (2020) Fully Convolutional Neural Network for Rapid Flood Segmentation in Synthetic Aperture Radar Imagery, MDPI. publisher. Available at: https://www.mdpi.com/2072-4292/12/16/2532 (Accessed: 11 December 2025).

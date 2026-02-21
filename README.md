GSA-LN

Experimental implementation of the GSA-LN architecture, exploring hybrid spatial-channel-frequency attention mechanisms for next-generation super-resolution

![GSA-LN Architecture](figs/arch.png)

## Dependencies

Based on `requirements.txt`, the following 3 main dependencies are used:
- **Python**: version 3.10
- **PyTorch**: Main deep learning framework (`torch` and `torchvision` dependencies).
- **Nvidia GPU + CUDA**: Highly recommended for GPU-accelerated computing.

## Pre-trained Models

You can download the pre-trained models for x2, x3, and x4 scales from [Google Drive](https://drive.google.com/drive/folders/1vvU5jfS7mScbvk8_JOIct8rfXY3VLwPd?usp=sharing).

## Datasets

| Type      | Dataset                                                                                             |
|-----------|-----------------------------------------------------------------------------------------------------|
| Training  | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](https://www.kaggle.com/datasets/daehoyang/flickr2k) |
| Testing   | [Set5 & Set14](https://www.kaggle.com/datasets/ll01dm/set-5-14-super-resolution-dataset), [BSD100](https://www.kaggle.com/datasets/asilva1691/bsd100), [Urban100](https://www.kaggle.com/datasets/harshraone/urban100) |

## Performance

| Model | Scale | Params | Set5 | Set14 | BSD100 | Urban100 | Manga109 |
|---|---|---|---|---|---|---|---|
| GSA-LN | 2 | 2.6M | 38.14 | 33.78 | 32.26 | 32.56 | 39.19 |
| GSA-LN | 3 | 2.8M | 34.56 | 30.47 | 29.18 | 28.52 | 34.12 |
| GSA-LN | 4 | 3.1M | 32.34 | 28.73 | 27.65 | 26.33 | 30.89 |

---
This project is built on top of [BasicSR](https://github.com/XPixelGroup/BasicSR)
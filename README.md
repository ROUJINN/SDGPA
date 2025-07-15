# SDGPA

Official implementation of paper: **Zero Shot Domain Adaptive Semantic Segmentation by Synthetic Data Generation and Progressive Adaptation**.(IROS 25')

by Jun Luo, Zijing Zhao, Yang Liu

# Installation

environment setting:

All of our experiments conducted on NVIDIA RTX 3090 with cuda 11.8
```
source env.sh
```
# Running

You can find all the training scripts in the `scripts/` folder

We use day $\to$ snow setting as an example.

First, you should decide where you want to put the datasets. Let's denote it as <data_root> (for example:`/data3/roujin`). By default, the experimental logs are stored in <data_root>.

Then, organize the folder as follows:
```
<data_root>
└─ ACDC
   └─ gt
   └─ rgb_anon
└─ cityscapes
   └─ gtFine
   └─ leftImg8bit
└─ GTA5
   └─ images
   └─ labels
```

You can refer to cityscapes and ACDC's official websites for the datasets. For GTA5, as we only use a subset of them, we provide the following links for your convenience. https://huggingface.co/datasets/roujin/GTA5subset

for synthetic data generation:
```
source img_gen/run.sh <data_root> snow
```

for progress model adaptation:
```
source scripts/snow.sh <data_root>
```

evaluation:
```
source eval.sh <data_root> <setting>
```
`<setting>` can be "day", "fog", "rain", "snow", "night", "game"

We release the following results. See all logs and checkpoints during training from https://huggingface.co/roujin/SDGPA/tree/main


| Setting          | Day→Night                                                                               | Clear→Snow                                                                             | Clear→Rain                                                                             | Clear→Fog                                                                             | Real→Game                                                                              |
| ---------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| results on paper | 26.9±0.8                                                                                | 47.4±0.7                                                                               | 48.6±0.8                                                                               | 58.8±0.7                                                                              | 43.4±0.4                                                                               |
| our released     | 27.6                                                                                    | 46.8                                                                                   | 49.0                                                                                   | 59.8                                                                                  | 43.1                                                                                   |
| checkpoint       | [link](https://huggingface.co/roujin/SDGPA/blob/main/night2/weights/weights_65.pth.tar) | [link](https://huggingface.co/roujin/SDGPA/blob/main/snow2/weights/weights_65.pth.tar) | [link](https://huggingface.co/roujin/SDGPA/blob/main/rain2/weights/weights_65.pth.tar) | [link](https://huggingface.co/roujin/SDGPA/blob/main/fog2/weights/weights_65.pth.tar) | [link](https://huggingface.co/roujin/SDGPA/blob/main/game2/weights/weights_65.pth.tar) |


We recommend you to read the scripts and the paper for more details.

For hyperparameter selection of InstructPix2Pix, we recommend reading:
https://huggingface.co/spaces/timbrooks/instruct-pix2pix/blob/main/README.md


# Acknowledgements

This code is built upon the following repositories:

https://github.com/azuma164/ZoDi

https://huggingface.co/timbrooks/instruct-pix2pix

We thank them for their excellent work!

# Citation

```
@inproceedings{luo2025sdgpa,
  title={Zero Shot Domain Adaptive Semantic Segmentation by Synthetic Data Generation and Progressive Adaptation},
  author={Luo, Jun and Zhao, Zijing and Liu, Yang},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  organization={IEEE}
}
```
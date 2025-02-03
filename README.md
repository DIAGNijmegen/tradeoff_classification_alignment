
# A Balancing Act: Optimizing Classification and Retrieval in Cross-Modal Vision Models

This repository includes the implementation of a simple, yet effective method to balance classification and contrastive objectives in general computer vision and computational Pathology. The paper is currently under review at MIDL 2025.


## Abstract
Despite the promising capabilities of vision-language models (VLMs) across diverse tasks, recent studies reveal that they struggle with the fundamental task of image classification. In this study, we explore leveraging state-of-the-art task-specific classification models as a foundation for VLMs, aiming to preserve strong classification performance. Specifically, we assess the impact of contrastive tuning to enable cross-modal retrieval capabilities on a Hierarchical Image Pyramid Transformer (HIPT) trained for prostate cancer grading in Whole-Slide Images (WSIs) and a ViT-Base model trained for multi-label classification on natural images. Our results demonstrate that contrastive fine-tuning creates a clear trade-off: classification accuracy rapidly deteriorates toward zero as vision-text alignment improves. By balancing the two objectives in the loss function during fine-tuning, we achieve competitive slide-level retrieval performance while maintaining classification accuracy.

## Requirements

- python 3.9+
- install requirements via `pip3 install -r requirements.txt`

# Code


**[Optional] Configure wandb**

If you want to benefit from wandb logging, you need to follow these simple steps:
 - grab your wandb API key under your wandb profile
 - run the following command in your terminal: `export WANDB_API_KEY=<your_personal_key>`
 - update wandb parameters in the relevant `.yaml` config files (see next)

# COCO experiments

**1.Download COCO 2014 dataset**
Download the 2014 Train/Val images + annotations with the below links and store it under `data/`

```
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

**2.[Optional] Create folds**

Create new folds from the COCO dataset:
```bash
python3 src/coco/preprocess.py
```

If you want to replicate my experiments, you can skip this step and use the folds saved under `data/folds/`

**2. Adapt config file**

Create a `.yaml` configuration file under `config/`. You can take inspiration from existing files `coco.yaml` for the coco experiments and `medical.yaml` for the medical experiments

**3. Contrastive Tuning using single $\lambda$ on a single fold**
```bash
python3 src/run.py --config-name {config_file_name}.yaml
```

Alternatively, you can adjust paramters dynamically e.g. in a bash script:

```bash
python3 src/run.py --config-name coco data.fold=$fold lambda_param=$lambda data.train_path="./data/folds/train_fold$fold.json" 
```

# Prostate cancer grading experiments
🚧 **Under Construction** 🚧  

## Prerequisite
This work uses a pretrained HIPT vision encoder that achieves state-of-the-art performance in multi-class ISUP grade classification with a quadratic kappa score of $0.892$ on the PANDA test set [1]. You can download the weights for this model: [here]

**1. Patch extraction**

 You need to extract square regions from each WSI (patches) you intend to train on. To do so, in this work I used the package: [HS2P](https://github.com/clemsgrs/hs2p) [2] which segments tissue and extract relevant patches at a given pixel spacing.

**2. Feature extraction**


**3. Adapt config file**


**4. Contrastive Tuning using single $\lambda$ on a single fold**




# References
## References

[1] C. Grisi, "HS2P: Histopathology Slide Pre-processing Pipeline", Available at: [https://github.com/clemsgrs/hs2p](https://github.com/clemsgrs/hs2p)

[2] C. Grisi, G. Litjens, and J. van der Laak, "Hierarchical Vision Transformers for Context-Aware Prostate Cancer Grading in Whole Slide Images," arXiv, December 2023. Available at: [https://arxiv.org/abs/2312.12619](https://arxiv.org/abs/2312.12619)

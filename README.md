
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
We pretrained a Local H-ViT model, leveraging a frozen patch-level Transformer, while finetuning region-level and slide-level Transformers for multi-class ISUP Grade classification. 
For implementation details, please refer to the [HIPT](https://github.com/clemsgrs/hipt) implementation. Train a single fold on the extracted features using the configuration in `config/hipt_train_panda.yaml.` The final model achieves state-of-the-art performance on the PANDA test set with a quadratic kappa score of $0.892$. 

To use this model during contrastive tuning, you need to:

1. **Patch Extraction:** 

    Extract $2048×2048$ patches from whole-slide images (WSIs) using [HS2P](https://github.com/clemsgrs/hs2p).
    Patches are sampled at $0.5$ pixel spacing to capture relevant tissue regions.
2. **Feature Extraction:** 
    Use a pretrained model to extract local features for each WSI, stored as tensors of shape (M, 64, 384). (See [HIPT](https://github.com/clemsgrs/hipt) for details).
3. **Adapt config file:**  
   Download the pretrained HIPT model weights (LocalGlobalHIPT_2048_768.pt) and update the `config/medical.yaml`. Set `vision.model_weights` to the path where the weights are strored and `vision.local_features_dir` to the output directory for feature extraction. This enables the model to generate 768-dim slide-level embeddings for prostate slides.

## Run Contrastive Tuning using $\lambda$ on a single fold: 
```bash
python3 src/run.py --config-name medical
```



# References
[1] C. Grisi, "Hierarchical Image Pyramid Transformer", Available at: [https://github.com/clemsgrs/hipt]https://github.com/clemsgrs/hipt)

[2] C. Grisi, "HS2P: Histopathology Slide Pre-processing Pipeline", Available at: [https://github.com/clemsgrs/hs2p](https://github.com/clemsgrs/hs2p)

<!-- 
[2] C. Grisi, G. Litjens, and J. van der Laak, "Hierarchical Vision Transformers for Context-Aware Prostate Cancer Grading in Whole Slide Images," arXiv, December 2023. Available at: [https://arxiv.org/abs/2312.12619](https://arxiv.org/abs/2312.12619) -->


# A Balancing Act: Optimizing Classification and Retrieval in Cross-Modal Vision Models

This repository includes the implementation of a simple, yet effective method to balance classification and contrastive objectives in general computer vision and computational Pathology. This paper was accepted in the full paper track at MIDL 2025.


## Abstract
Despite the promising capabilities of vision-language models (VLMs) in diverse tasks, recent studies reveal that they struggle with the fundamental task of image classification. In this study, we explore leveraging state-of-the-art task-specific classification models as a foundation for VLMs, aiming to preserve strong classification performance. Specifically, we assess the impact of contrastive tuning to enable cross-modal retrieval capabilities on a Vision Transformer (ViT) model trained for multi-label classification on natural images and a Hierarchical Vision Transformer (H-ViT) trained for prostate cancer grading in Whole-Slide Images (WSIs). Our results demonstrate that contrastive fine-tuning creates a clear trade-off: classification accuracy rapidly deteriorates toward zero as vision-text alignment improves. By balancing task-specific and contrastive objectives in the loss function during fine-tuning, we achieve competitive slide-level retrieval performance while maintaining classification accuracy. 

## Requirements

- python 3.9+
- install requirements via `pip3 install -r requirements.txt`

**Clone repository**
```bash
git clone https://github.com/DIAGNijmegen/tradeoff_classification_alignment.git
```

**Set PYTHONPATH in your terminal**
```bash
export PYTHONPATH="/absolute/path/to/classification-alignment-tradeoff"
```

**[Optional] Configure wandb**

If you want to benefit from wandb logging, you need to follow these simple steps:
 - grab your wandb API key under your wandb profile
 - run the following command in your terminal: `export WANDB_API_KEY=<your_personal_key>`
 - update wandb parameters in the relevant `.yaml` config files (see next)

# COCO experiments

**1. Download COCO 2017 dataset**

Download the 2017 Train/Val images and the corresponding captions (annotations) with the below script:

```bash
python3 download_coco_data.sh
```

**2. Prepare data**
To replicate the folds used in the experiment run:
```bash
python3 src/coco/preprocess.py
```

After running `src/preprocess.py`, your data directory will look like this:
```
data/
├── coco/
│   ├── testset.json     # independent test set
│   └── folds/           # 5-fold cross-validation splits
│       ├── train_fold0.json
│       ├── ...
│       ├── val_fold0.json
│       ├── ...
              
```
**2. Adapt config file**

Create a `.yaml` configuration file under `config/`. You can take inspiration from existing files `coco.yaml` for the coco experiments and `medical.yaml` for the medical experiments

**3. Contrastive Tuning using single $\lambda$ on a single fold**
```bash
python3 src/run.py --config-name coco.yaml
```

Alternatively, you can adjust paramters dynamically e.g. in a bash script:

```bash
python3 src/run.py --config-name coco data.fold=$fold lambda_param=$lambda data.train_path="./data/folds/train_fold$fold.json" 
```

# Prostate cancer grading experiments


## Prerequisite
We pretrained a Local H-ViT model, leveraging a frozen patch-level Transformer, while finetuning region-level and slide-level Transformers for multi-class ISUP Grade classification on the PANDA dataset. 
For implementation details, please refer to the [HIPT](https://github.com/clemsgrs/hipt) implementation. Train a single fold on the extracted features using the configuration in `config/hipt_panda.yaml.` The final model achieves state-of-the-art performance on the PANDA test set with a quadratic kappa score of $0.892$. 

To use this model during contrastive tuning, you need to:
**Clone HIPT repository**
```bash
git clone https://github.com/clemsgrs/hipt.git

```
**Set PYTHONPATH**
```bash
export PYTHONPATH="/absolute/path/to/classification-alignment-tradeoff:/absolute/path/to/hipt"
```

## Prepare data ##

Similar as above, the data directory should be organized like this:
```
data/
├── medical/
│   ├── testset.json     # independent test set
│   └── folds/           # 5-fold cross-validation splits
│       ├── train_fold0.json
│       ├── ...
│       ├── val_fold0.json
│       ├── ...           
```
### JSON File Format

Each `train_foldX.json`, `val_foldX.json` and `testset_json.json` the WSI-report pairs. 
Each case is a dictionary with the following structure:

```json
{
  {
    "case_id": "uid0",
    "label": 0,
    "report": "Microscopy: ... Conclusion: ..."
  },
  {
    "case_id": "uidX",
    "label": 4,
    "report": "Microscopy: ... Conclusion: ..."
  }
}
```
Here, `label` refers to the ISUP grade (ranging from 0 to 5), where `0` indicates a benign case and `1–5` represent increasing grades of malignancy.  
The `report` field contains the full diagnostic report, which combines both the *microscopy* and *conclusion* sections.  
The `case_id` corresponds to the unique identifier (`uid`) of the associated Whole-Slide Image.

## Preprocessing Whole-Slide-Images
1. **Patch Extraction:** 

    Extract $2048×2048$ patches from whole-slide images (WSIs) using [HS2P](https://github.com/clemsgrs/hs2p).
    Patches are sampled at $0.5$ pixel spacing to capture relevant tissue regions.
2. **Feature Extraction:** 

    Use a pretrained model to extract local features for each WSI, stored as tensors of shape (M, 64, 384). (See [HIPT](https://github.com/clemsgrs/hipt) for details).
3. **Adapt config file:**  

   Download the pretrained HIPT model weights (LocalGlobalHIPT_2048_768.pt) and update the `config/medical.yaml`. Set `vision.model_weights` to the path where the weights are strored and `vision.local_features_dir` to the output directory for feature extraction. This enables the model to generate 768-dim slide-level embeddings for prostate slides.


## Run Contrastive Tuning using $\lambda$ on a single fold: 
```bash
python3 src/run.py --config-name medical lambda_param=0.9 lambda_param=0
```



# References
[1] C. Grisi, "Hierarchical Image Pyramid Transformer", Available at: [https://github.com/clemsgrs/hipt]https://github.com/clemsgrs/hipt)

[2] C. Grisi, "HS2P: Histopathology Slide Pre-processing Pipeline", Available at: [https://github.com/clemsgrs/hs2p](https://github.com/clemsgrs/hs2p)



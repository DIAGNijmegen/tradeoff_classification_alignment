
# A Balancing Act: Optimizing Classification and Retrieval in Cross-Modal Vision Models

This repository includes the implementation of a simple, yet effective method to balance classification and contrastive objectives in general computer vision and computational Pathology. The paper is currently under review at MIDL 2025.


## Abstract
Despite the promising capabilities of vision-language models (VLMs) across diverse tasks, recent studies reveal that they struggle with the fundamental task of image classification. In this study, we explore leveraging state-of-the-art task-specific classification models as a foundation for VLMs, aiming to preserve strong classification performance. Specifically, we assess the impact of contrastive tuning to enable cross-modal retrieval capabilities on a Hierarchical Image Pyramid Transformer (HIPT) trained for prostate cancer grading in Whole-Slide Images (WSIs) and a ViT-Base model trained for multi-label classification on natural images. Our results demonstrate that contrastive fine-tuning creates a clear trade-off: classification accuracy rapidly deteriorates toward zero as vision-text alignment improves. By balancing the two objectives in the loss function during fine-tuning, we achieve competitive slide-level retrieval performance while maintaining classification accuracy.




# Code
🚧 **Under Construction** 🚧  

To run the experiments on the COCO task with all metrics and the default dictionary:
```bash
python src/run_coco.py --config-name coco lambda_param=1.0 
```
To run the prostate cancer grading experiments:
```bash
python src/run_coco.py --config-name medical lambda_param=1.0
```
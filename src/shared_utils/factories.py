import os
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from transformers import (
    RobertaConfig,
    BertModel,
    ViTFeatureExtractor,
    ViTForImageClassification,
)
from sentence_transformers import SentenceTransformer

from shared_utils.models import (
    RobertaSequenceClassification,
    ViTForImageClassificationWithEmbeddings,
)
from shared_utils.losses import CLIPLoss
from source.models import LocalGlobalRegressionHIPT #comes from the HIPT repository


class LossFactory:
    def __init__(
        self,
        loss_name: str,
        margin: Optional[float] = 1.0,
    ) -> None:
        """
        Factory to return loss functions based on a string identifier.

        Args:
            loss_name (str): Name of the loss function.
            margin (Optional[float]): Used for margin-based losses like Triplet.

        Raises:
            KeyError: If the loss name is not supported.
        """
        loss_mapping: Dict[str, nn.Module] = {
            "triplet": nn.TripletMarginLoss(margin=margin, p=2),
            "mse": nn.MSELoss(),
            "mean-squared-error": nn.MSELoss(),
            "ce": nn.CrossEntropyLoss(),
            "cross-entropy": nn.CrossEntropyLoss(),
            "bce": nn.BCEWithLogitsLoss(),
            "binary-cross-entropy": nn.BCEWithLogitsLoss(),
            "clip": CLIPLoss(),
        }

        if loss_name in loss_mapping:
            self.criterion = loss_mapping[loss_name]
        else:
            raise KeyError(f"{loss_name} not supported")

    def get_loss(self) -> nn.Module:
        """
        Returns:
            torch.nn.Module: The loss function.
        """
        return self.criterion
    

class OptimizerFactory:
    def __init__(
        self,
        name: str,
        params: Any,
        lr: float,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ) -> None:
        """
        Creates an optimizer based on the specified name and parameters.

        Args:
            name (str): Optimizer name ("adam", "adamw", or "sgd").
            params (iterable): Model parameters to optimize.
            lr (float): Learning rate.
            weight_decay (float): Weight decay (L2 regularization).
            momentum (float): Momentum (used for SGD).
        """
        if name == "adam":
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif name == "adamw":
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif name == "sgd":
            self.optimizer = optim.SGD(
                params, lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise KeyError(f"{name} not supported")
    def get_optimizer(self) -> optim.Optimizer:
        """Returns the created optimizer."""
        return self.optimizer
    
class ModelFactory:
    def __init__(self, model_type: str, cfg: DictConfig) -> None:
        """
        Factory that returns the correct model instance based on the model_type.

        Args:
            model_type (str): Type of model to load.
            cfg (DictConfig): Configuration with model parameters.

        Raises:
            KeyError: If model_type is not supported.
        """
        self.model = None

        if model_type == "LocalGlobalRegressionHIPT":
            vision_config = OmegaConf.load(cfg.vision.config)
            model_options = vision_config.model
            self.model = LocalGlobalRegressionHIPT(
                region_size=model_options.region_size,
                patch_size=model_options.patch_size,
                embed_dim_slide=model_options.embed_dim_slide,
                pretrain_vit_region=model_options.pretrain_vit_region,
                freeze_vit_region=model_options.freeze_vit_region,
                freeze_vit_region_pos_embed=model_options.freeze_vit_region_pos_embed,
                dropout=model_options.dropout,
                slide_pos_embed=model_options.slide_pos_embed,
                mask_attn_region=model_options.mask_attn_region,
                img_size_pretrained=model_options.img_size_pretrained,
            )
            self.model.load_state_dict(
                torch.load(cfg.vision.model_weights, map_location='cuda', weights_only=True)
            )

        elif model_type == "RobertaSequenceClassification":
            config = RobertaConfig.from_pretrained(cfg.language.base_dir)
            self.model = RobertaSequenceClassification(
                base_model_path=config._name_or_path,
                num_labels=config.num_labels,
                freeze_base_params=True
            )
            self.model.load_state_dict(
                torch.load(os.path.join(cfg.language.base_dir, "model_state_dict.pt"), weights_only=True)
            )

        elif model_type == "bert-uncased":
            self.model = BertModel.from_pretrained('bert-base-uncased')

        elif model_type == "vit_base":
            feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            vit_model.classifier = nn.Linear(vit_model.config.hidden_size, cfg.num_classes)

            self.model = ViTForImageClassificationWithEmbeddings(vit_model)

            if cfg.vision.finetuned_vit_path is not None:
                print('Loading finetuned ViT model')
                self.model.load_state_dict(
                    torch.load(cfg.vision.finetuned_vit_path, map_location='cuda', weights_only=True)
                )

            if cfg.vision.freeze_lower_layers:
                print('Freezing first 5 layers of VitBase')
                for i, layer in enumerate(vit_model.vit.encoder.layer):
                    for param in layer.parameters():
                        param.requires_grad = i >= 4  # Freeze first 4 layers only

        elif model_type in ['sentenceBert', 'mpnet', 'sentenceRoBerta', 'biobert']:
            pretrained_map = {
                'sentenceBert': 'all-MiniLM-L6-v2',
                'mpnet': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
                'sentenceRoBerta': 'roberta-base-nli-stsb-mean-tokens',
                'biobert': 'pritamdeka/S-BioBert-snli-multinli-stsb'
            }
            self.model = SentenceTransformer(pretrained_map[model_type])

        else:
            raise KeyError(f"{model_type} not supported")

    def get_model(self):
        return self.model



# class ModelFactory:
#     def __init__(self, model_type: str, cfg: DictConfig) -> None:
#         """Model factory that returns correct model given model_type
#         Args:
#             model_type (str): _description_
#             num_classes (Optional[int], optional): _description_. Defaults to None.
#             label_encoding (Optional[str], optional): _description_. Defaults to None.
#             model_options (Optional[DictConfig], optional): _description_. Defaults to None.
#             base_model_path (Optional[str], optional): _description_. Defaults to None.
#             freeze_base_params (Optional[bool], optional): _description_. Defaults to None.

#         Raises:
#             KeyError: _description_
#         """
#         if model_type == "LocalGlobalRegressionHIPT":
#             vision_config = OmegaConf.load(cfg.vision.config)
#             model_options = vision_config.model
#             self.model = LocalGlobalRegressionHIPT(
#                 region_size=model_options.region_size,
#                 patch_size=model_options.patch_size,
#                 embed_dim_slide=model_options.embed_dim_slide,  # added by judith
#                 pretrain_vit_region=model_options.pretrain_vit_region,
#                 freeze_vit_region=model_options.freeze_vit_region,
#                 freeze_vit_region_pos_embed=model_options.freeze_vit_region_pos_embed,
#                 dropout=model_options.dropout,
#                 slide_pos_embed=model_options.slide_pos_embed,
#                 mask_attn_region=model_options.mask_attn_region,
#                 img_size_pretrained=model_options.img_size_pretrained,
#             )

#             self.model.load_state_dict(torch.load(cfg.vision.model_weights, map_location='cuda',weights_only=True))

#         elif model_type == "RobertaSequenceClassification":
#             config = RobertaConfig.from_pretrained(cfg.language.base_dir)

#             self.model = RobertaSequenceClassification(
#                 base_model_path=config._name_or_path, num_labels = config.num_labels, freeze_base_params=True
#             )
#             self.model.load_state_dict(torch.load(os.path.join(cfg.language.base_dir, "model_state_dict.pt"),weights_only=True))

#         elif model_type =="bert-uncased":
#             # Load pretrained BERT model (frozen)
#             self.model = BertModel.from_pretrained('bert-base-uncased')
#         elif model_type=='vit_base':
#             # Load the feature extractor and pretrained ViT model
#             feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
#             vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
           
#             # Modify the classification head to match the number of COCO classes (80)
#             vit_model.classifier = nn.Linear(vit_model.config.hidden_size, cfg.num_classes)
#             self.model = ViTForImageClassificationWithEmbeddings(vit_model)
#             if cfg.vision.finetuned_vit_path is not None:
#                 print('Loading finetuned ViT model')
#                 self.model.load_state_dict(torch.load(cfg.vision.finetuned_vit_path, map_location='cuda',weights_only=True))
#             if cfg.vision.freeze_lower_layers:
#                 print('Freezing first 5 layers of VitBase')
#                 for i, layer in enumerate(vit_model.vit.encoder.layer):
#                     if i < 4:  # Freeze only the first 4 layers
#                         for param in layer.parameters():
#                             param.requires_grad = False
#                     else:
#                         for param in layer.parameters():
#                             param.requires_grad = True  # Explicitly unfreeze other layers
                
                
#         elif model_type=='sentenceBert':
#             self.model = SentenceTransformer('all-MiniLM-L6-v2') 
#         elif model_type=='mpnet':
#             self.model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
#         elif model_type=='sentenceRoBerta':
#              self.model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
#         elif model_type=='biobert':
#             self.model = SentenceTransformer('pritamdeka/S-BioBert-snli-multinli-stsb' ) 

#         else:
#             raise KeyError(f"{model_type} not supported")

#     def get_model(self):
#         return self.model

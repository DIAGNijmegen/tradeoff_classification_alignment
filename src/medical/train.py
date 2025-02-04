import os
import torch
import tqdm
import numpy as np
import torch.nn as nn
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from transformers.utils.logging import disable_progress_bar

# Disable the progress bar globally
disable_progress_bar()
import warnings

# Suppress sklearn UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)
from src.shared_utils import normalize_embeddings,compute_losses,save_checkpoint
from src.medical.utils import (
    get_metrics,
    get_preds_from_regression_logits,
    retrieval_kappa,
    image_retrieval
)

def get_report_embeddings(language_model, report_tokens, device):
    """
    Generates report embeddings based on the type of language model.

    Args:
        language_model: The language model (SentenceTransformer, CoCa, or others).
        report_tokens: The tokenized reports.
        device: The device to move embeddings to.

    Returns:
        torch.Tensor: The computed report embeddings.
    """
    if isinstance(language_model, SentenceTransformer):  # or isinstance(language_model, CoCa)
        embeddings = language_model.encode(report_tokens, show_progress_bar=False)
        return torch.from_numpy(embeddings).to(device)
    
    # For other models that require token dictionaries
    report_tokens = {key: val.squeeze(1).to(device) for key, val in report_tokens.items()}
    _, embeddings = language_model(**report_tokens)
    return embeddings.to(device)



def train_epoch(
    epoch: int,
    dataset: torch.utils.data.Dataset,
    vision_model: nn.Module,
    language_model: nn.Module,
    optimizer_vision: optim.Optimizer,
    classification_criterion: nn.Module,
    contrastive_criterion: nn.Module,
    checkpoint_dir: str,
    cfg: Dict,
) -> Tuple[Dict, str]:
    """
    Trains the vision model for one epoch.

    Args:
        epoch: Current epoch number.
        dataset: Training dataset.
        vision_model: Vision model to train.
        language_model: Language model for encoding reports.
        optimizer_vision: Optimizer for vision model.
        classification_criterion: Classification loss function.
        contrastive_criterion: Contrastive loss function.
        checkpoint_dir: Directory to save model checkpoints.
        cfg: Configuration dictionary.

    Returns:
        A dictionary of training metrics and the checkpoint path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(
        dataset, batch_size=cfg.training.batch_size, shuffle=True, pin_memory=True
    )

    vision_model.train()
    language_model.eval()

    image_embeddings_epoch, report_embeddings_epoch = [], []
    predictions, labels, probabilities_list = [], [], []
    epoch_loss, epoch_contrastive_loss, epoch_classification_loss = 0, 0, 0

    for image, positive_report_tokens, negative_report_tokens, label in tqdm.tqdm(
        data_loader, desc=f"Epoch {epoch} - Train", unit="batch", ncols=80, leave=False
    ):

        feature = torch.load(image[0], map_location=device,weights_only=True)
        logits, image_embedding = vision_model(feature)

        probabilities = F.softmax(logits, dim=-1)
        probabilities_list.append(probabilities.detach().cpu())
        preds = get_preds_from_regression_logits(
                logits.cpu(), cfg.num_classes
            )[0]
        labels.extend(label.cpu().tolist())
        predictions.extend(preds.clone().tolist())

        #get text embeddings
        pos_report_embedding = get_report_embeddings(language_model, positive_report_tokens, device)
        neg_report_embedding = get_report_embeddings(language_model, negative_report_tokens, device)

        #normalize embeddings
        image_embedding, pos_report_embedding, neg_report_embedding = normalize_embeddings(
            image_embedding, pos_report_embedding, neg_report_embedding
        )
        image_embeddings_epoch.append(image_embedding)
        report_embeddings_epoch.append(pos_report_embedding)

        loss, classification_loss, contrastive_loss = compute_losses(
            logits,
            label,
            image_embedding,
            pos_report_embedding,
            neg_report_embedding,
            classification_criterion,
            contrastive_criterion,
            cfg,
        )

        epoch_loss += loss.item()
        epoch_contrastive_loss += contrastive_loss.item()
        epoch_classification_loss += classification_loss.item()

        (
            (loss / cfg.training.gradient_accumulation_steps).backward()
            if cfg.training.gradient_accumulation_steps
            else loss.backward()
        )

        if not cfg.training.gradient_accumulation_steps or (
            len(predictions) % cfg.training.gradient_accumulation_steps == 0
        ):
            optimizer_vision.step()
            optimizer_vision.zero_grad()

        torch.cuda.empty_cache()

    checkpoint_path = save_checkpoint(
        epoch, vision_model, optimizer_vision, epoch_loss, checkpoint_dir
    )

    results = get_metrics(
        preds=predictions,
        labels=labels,
        class_names=[f"isup_{i}" for i in range(cfg.num_classes)],
        use_wandb=cfg.wandb.enable,
    )
    results.update(
        {
            "combined loss": epoch_loss / len(data_loader),
            "contrastive loss": epoch_contrastive_loss / len(data_loader),
            "classification loss": epoch_classification_loss / len(data_loader),
            "mean cosine similarity": nn.CosineSimilarity(dim=1, eps=1e-6)(
                torch.vstack(image_embeddings_epoch),
                torch.vstack(report_embeddings_epoch),
            )
            .mean()
            .item(),
        }
    )

    print(f"Epoch [{epoch}] - Train Loss: {results['combined loss']:.4f}")
    return results, checkpoint_path


def tune_epoch(
    epoch: int,
    dataset: torch.utils.data.Dataset,
    vision_model: nn.Module,
    language_model: nn.Module,
    classification_criterion: nn.Module,
    contrastive_criterion: nn.Module,
    cfg: Dict,  # Non-default argument should come before default arguments
    phase: str = "validation",  # Default argument
) -> Dict:
    """
    Evaluates the vision model on a dataset for one epoch.

    Args:
        epoch: Current epoch number.
        dataset: Evaluation dataset.
        vision_model: Vision model to evaluate.
        language_model: Language model for encoding reports.
        classification_criterion: Classification loss function.
        contrastive_criterion: Contrastive loss function.
        phase: Specifies whether this is validation or another phase.
        cfg: Configuration dictionary.

    Returns:
        A dictionary of evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)

    vision_model.eval()
    language_model.eval()

    image_embeddings_epoch, report_embeddings_epoch = [], []
    predictions, labels, probabilities_list, original_captions = [], [], [], []
    epoch_loss, epoch_contrastive_loss, epoch_classification_loss = 0, 0, 0

    with torch.no_grad():
        for image, positive_report_tokens, negative_report_tokens, label in tqdm.tqdm(
            data_loader,
            desc=f"Epoch {epoch} - Tune",
            unit="batch",
            ncols=80,
            leave=False,
        ):
            #get image embedding
            feature = torch.load(image[0], map_location=device, weights_only=True)
            logits, image_embedding = vision_model(feature)
            
            logits, label, image_embedding = (
                    logits.to(device,non_blocking=True),
                    label.to(device,non_blocking=True),
                    image_embedding.to(device,non_blocking=True),
                )
            probabilities = F.softmax(logits, dim=-1)
            probabilities_list.append(probabilities.detach().cpu())
            preds = get_preds_from_regression_logits(
                    logits.cpu(), cfg.num_classes
                )[0]
            labels.extend(label.cpu().tolist())
            predictions.extend(preds.clone().tolist())
            original_captions.extend(list(positive_report_tokens))

            #get text embeddings
            pos_report_embedding = get_report_embeddings(language_model, positive_report_tokens, device)
            neg_report_embedding = get_report_embeddings(language_model, negative_report_tokens, device)


            image_embedding, pos_report_embedding, neg_report_embedding = normalize_embeddings(
                image_embedding, pos_report_embedding, neg_report_embedding
            )
            image_embeddings_epoch.append(image_embedding)
            report_embeddings_epoch.append(pos_report_embedding)

            loss, classification_loss, contrastive_loss = compute_losses(
                logits,
                label,
                image_embedding,
                pos_report_embedding,
                neg_report_embedding,
                classification_criterion,
                contrastive_criterion,
                cfg,
            )

            epoch_loss += loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            epoch_classification_loss += classification_loss.item()
            torch.cuda.empty_cache()

    results = get_metrics(
            preds=predictions,
            labels=labels,
            class_names=[f"isup_{i}" for i in range(cfg.num_classes)],
            use_wandb=cfg.wandb.enable)
    results.update(
        {
            "combined loss": epoch_loss / len(data_loader),
            "contrastive loss": epoch_contrastive_loss / len(data_loader),
            "classification loss": epoch_classification_loss / len(data_loader),
            "mean cosine similarity": nn.CosineSimilarity(dim=1, eps=1e-6)(
                torch.vstack(image_embeddings_epoch),
                torch.vstack(report_embeddings_epoch),
            )
            .mean()
            .item(),
        }
    )
    cosine_sim_matrix = torch.mm(
        torch.vstack(image_embeddings_epoch), torch.vstack(report_embeddings_epoch).t()
    )
    results['retrieval kappa'], _, original_captions_test,_,image_ids= retrieval_kappa(cosine_sim_matrix,cfg.data.caption_column, cfg.data.fold, phase=phase, config=cfg)  # changed this frm 'validation' to phase 

    top_k_accuracy, _, original_captions, _, _ = image_retrieval(
        cosine_sim_matrix, 
        cfg.data.caption_column, 
        phase=phase, #chnaged from validation to phase
        top_k_values=cfg.training.topk,
        config=cfg
    )
    for k in cfg.training.topk:
            results[f'retrieval top_{k}'] = round(top_k_accuracy[k],3)
   
    print(f"Epoch [{epoch}] - Tune Loss: {results['combined loss']:.4f}")
    return results


def test(
    dataset: torch.utils.data.Dataset,
    vision_model: nn.Module,
    language_model: nn.Module,
    wandb_instance=None,
    save_results_dir: Optional[str] = None,
    cfg: Dict = None,
    phase: Optional[str] = "test",
) -> Dict:
    """
    Evaluates the vision model on a test dataset.

    Args:
        dataset: Test dataset.
        vision_model: Vision model to evaluate.
        language_model: Language model for encoding reports.
        wandb_instance: Optional Weights & Biases instance for logging.
        save_results_dir: Directory to save retrieval results.
        phase: Specifies the testing phase.
        cfg: Configuration dictionary.

    Returns:
        A dictionary of test metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False)

    vision_model.eval()
    language_model.eval()

    image_embeddings, report_embeddings = [], []
    predictions, labels, probabilities_list, original_captions = [], [], [], []

    with torch.inference_mode():
        for image, positive_report_tokens, _, label in tqdm.tqdm(
            data_loader, desc="Test", unit="batch", ncols=80, leave=False
        ):
            feature = torch.load(image[0], map_location=device,weights_only=True)
            logits, image_embedding = vision_model(feature)
            # Move to device
            logits, label, image_embedding = (
                    logits.to(device),
                    label.to(device),
                    image_embedding.to(device),
                )
            probabilities = F.softmax(logits, dim=-1)
            probabilities_list.append(probabilities)
            preds = torch.argmax(probabilities, dim=-1)
            predictions.extend(preds.clone().tolist())
            labels.extend(label.cpu().tolist())

            original_captions.extend(list(positive_report_tokens))
            #get text embeddings
            pos_report_embedding = get_report_embeddings(language_model, positive_report_tokens, device)
          
            image_embedding, pos_report_embedding = normalize_embeddings(
                image_embedding, pos_report_embedding
            )
            image_embeddings.append(image_embedding)
            report_embeddings.append(pos_report_embedding)
            torch.cuda.empty_cache()

    results = get_metrics(
            preds=predictions,
            labels=labels,
            class_names=[f"isup_{i}" for i in range(cfg.num_classes)],
            use_wandb=cfg.wandb.enable,
        )
    image_embeddings = torch.vstack(image_embeddings)
    report_embeddings = torch.vstack(report_embeddings)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    results["mean cosine similarity"] = (
        cos(image_embeddings, report_embeddings).mean().item()
    )
    cosine_sim_matrix = torch.mm(image_embeddings, report_embeddings.t())
    results['retrieval kappa'], retrieval_status, original_captions_test,retrieved_captions,image_ids= retrieval_kappa(cosine_sim_matrix, cfg.training.caption_column, cfg.data.fold, phase=phase, config=cfg)
    top_k_accuracy, _, _, _, _ = image_retrieval(
    cosine_sim_matrix, 
    cfg.training.caption_column, 
    phase=phase, 
    top_k_values=cfg.training.topk,config = cfg)
    for k in cfg.training.topk:
        results[f'retrieval top_{k}'] = top_k_accuracy[k]


  
    if cfg.wandb.enable:
        table = wandb_instance.Table(columns=["Slide id", "Original Report", f"Top {1} Retrieved Report", "Correct label", "Label"])
        for img_id, orig_caption, topk_caption,status, label in zip(image_ids, original_captions_test, retrieved_captions, retrieval_status, labels):
    
                    table.add_data(img_id, orig_caption, topk_caption, status, label) 
                    wandb_instance.log({"Image-Caption-Retrieval Test": table})

    return results
import os
import torch
import tqdm
import wandb
import numpy as np
import torch.nn as nn
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple

from transformers.utils.logging import disable_progress_bar

# Disable the progress bar globally
disable_progress_bar()
import warnings

# Suppress sklearn UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UserWarning)
from src.shared_utils import normalize_embeddings,compute_losses,save_checkpoint
from src.coco.utils import (
    get_metrics_coco,
    coco_retrieval,
)


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

    for image, pos_caption, neg_captions, label in tqdm.tqdm(
        data_loader, desc=f"Epoch {epoch} - Train", unit="batch", ncols=80, leave=False
    ):
        image, label = image.to(device), label.to(device)
        logits, image_embedding = vision_model(image)
        probabilities = torch.sigmoid(logits).detach().cpu()
        predictions.extend((probabilities > 0.5).int().tolist())
        labels.extend(label.cpu().tolist())
        probabilities_list.append(probabilities)

        with torch.no_grad():
            pos_embedding = torch.from_numpy(
                language_model.encode(pos_caption, show_progress_bar=False)
            ).to(device)
            neg_embedding = torch.from_numpy(
                language_model.encode(neg_captions, show_progress_bar=False)
            ).to(device)

        image_embedding, pos_embedding, neg_embedding = normalize_embeddings(
            image_embedding, pos_embedding, neg_embedding
        )
        image_embeddings_epoch.append(image_embedding)
        report_embeddings_epoch.append(pos_embedding)

        loss, classification_loss, contrastive_loss = compute_losses(
            logits,
            label,
            image_embedding,
            pos_embedding,
            neg_embedding,
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

    epoch_probabilities = torch.vstack(probabilities_list)
    results = get_metrics_coco(
        preds=predictions,
        probs=epoch_probabilities,
        labels=labels,
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
        for image, pos_caption, neg_captions, label in tqdm.tqdm(
            data_loader,
            desc=f"Epoch {epoch} - Tune",
            unit="batch",
            ncols=80,
            leave=False,
        ):
            image, label = image.to(device, non_blocking=True), label.to(
                device, non_blocking=True
            )
            logits, image_embedding = vision_model(image)
            probabilities = torch.sigmoid(logits)
            probabilities_list.append(probabilities)
            predictions.extend((probabilities > 0.5).int().tolist())
            labels.extend(label.cpu().tolist())
            original_captions.extend(list(pos_caption))

            pos_embedding = torch.from_numpy(
                language_model.encode(pos_caption, show_progress_bar=False)
            ).to(device)
            neg_embedding = torch.from_numpy(
                language_model.encode(neg_captions, show_progress_bar=False)
            ).to(device)

            image_embedding, pos_embedding, neg_embedding = normalize_embeddings(
                image_embedding, pos_embedding, neg_embedding
            )
            image_embeddings_epoch.append(image_embedding)
            report_embeddings_epoch.append(pos_embedding)

            loss, classification_loss, contrastive_loss = compute_losses(
                logits,
                label,
                image_embedding,
                pos_embedding,
                neg_embedding,
                classification_criterion,
                contrastive_criterion,
                cfg,
            )

            epoch_loss += loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            epoch_classification_loss += classification_loss.item()
            torch.cuda.empty_cache()

    epoch_probabilities = torch.vstack(probabilities_list)
    results = get_metrics_coco(
        preds=predictions,
        labels=labels,
        probs=epoch_probabilities,
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

    cosine_sim_matrix = torch.mm(
        torch.vstack(image_embeddings_epoch), torch.vstack(report_embeddings_epoch).t()
    )
    for k in cfg.training.topk:
        results[f"coco retrieval top_{k}"], _, _, _ = coco_retrieval(
            cosine_sim_matrix, original_captions, k=k, phase=phase, config=cfg
        )

    print(f"Epoch [{epoch}] - Tune Loss: {results['combined loss']:.4f}")
    return results


def test(
    dataset: torch.utils.data.Dataset,
    vision_model: nn.Module,
    language_model: nn.Module,
    wandb_instance=None,
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
        for image, pos_caption, _, label in tqdm.tqdm(
            data_loader, desc="Test", unit="batch", ncols=80, leave=False
        ):
            image, label = image.to(device), label.to(device)
            logits, image_embedding = vision_model(image)
            probabilities = torch.sigmoid(logits)
            probabilities_list.append(probabilities)
            predictions.extend((probabilities > 0.5).int().tolist())
            labels.extend(label.cpu().tolist())

            original_captions.extend(list(pos_caption))
            pos_embedding = torch.from_numpy(
                language_model.encode(pos_caption, show_progress_bar=False)
            ).to(device)

            image_embedding, pos_embedding = normalize_embeddings(
                image_embedding, pos_embedding
            )
            image_embeddings.append(image_embedding)
            report_embeddings.append(pos_embedding)
            torch.cuda.empty_cache()

    epoch_probabilities = torch.vstack(probabilities_list)
    results = get_metrics_coco(
        preds=predictions,
        probs=epoch_probabilities,
        labels=labels,
        use_wandb=cfg.wandb.enable,
    )

    image_embeddings = torch.vstack(image_embeddings)
    report_embeddings = torch.vstack(report_embeddings)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    results["mean cosine similarity"] = (
        cos(image_embeddings, report_embeddings).mean().item()
    )

    cosine_sim_matrix = torch.mm(image_embeddings, report_embeddings.t())
    for k in cfg.training.topk:
        (
            results[f"coco retrieval top_{k}"],
            retrieval_status,
            retrieved_captions,
            all_original_captions,
        ) = coco_retrieval(
            cosine_sim_matrix, original_captions, k=k, phase="test", config=cfg
        )
        

    if cfg.wandb.enable:
        table = wandb.Table(
            columns=[
                "Original Caption",
                f"Top {k} Retrieved Captions",
                "Correct retrieval",
            ]
        )
        for orig_caption, topk_captions, status in zip(
            all_original_captions, retrieved_captions, retrieval_status
        ):
            table.add_data("\n".join(orig_caption), "\n".join(topk_captions), status)
        wandb_instance.log({"Image-Caption-Retrieval": table})

    return results

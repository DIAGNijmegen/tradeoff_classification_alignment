import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any


def compute_losses(
    logits: torch.Tensor,
    label: torch.Tensor,
    image_embedding: torch.Tensor,
    positive_report_embedding: torch.Tensor,
    negative_report_embedding: Optional[torch.Tensor],
    classification_criterion: nn.Module,
    contrastive_criterion: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes classification and contrastive losses.

    Args:
        logits: Model output logits.
        label: Ground truth labels.
        image_embedding: Embedding of input images.
        positive_report_embedding: Embedding of positive reports.
        negative_report_embedding: Embedding of negative reports.
        classification_criterion: Loss function for classification.
        contrastive_criterion: Loss function for contrastive learning.
        cfg: Configuration dictionary.

    Returns:
        Tuple of total loss, classification loss, and contrastive loss.
    """
    if cfg.experiment_name == 'medical':
        label = label.to(logits.device).float()
        classification_loss = classification_criterion(logits.squeeze(0), label.float())
    else:
        classification_loss = classification_criterion(logits, label.float())

    if isinstance(contrastive_criterion, nn.TripletMarginLoss):
        contrastive_loss = contrastive_criterion(
            image_embedding, positive_report_embedding, negative_report_embedding
        )
    else:
        contrastive_loss = contrastive_criterion(
            image_embedding, positive_report_embedding
        )

    loss = (
        cfg.lambda_param * contrastive_loss
        + (1.0 - cfg.lambda_param) * classification_loss
    )
    return loss, classification_loss, contrastive_loss


class CLIPLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))

    def forward(self, image_embeddings, text_embeddings):
        # Normalize the embeddings
        image_embeddings = normalize_embeddings(image_embeddings)[0]
        text_embeddings = normalize_embeddings(text_embeddings)[0]

        # Scale the logits by the temperature
        logit_scale = torch.exp(self.logit_scale)

        # Compute cosine similarity between image and text embeddings
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.t()) * logit_scale
        logits_per_text = logits_per_image.t()

        # Compute the cross-entropy loss
        batch_size = image_embeddings.shape[0]
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=image_embeddings.device)

        # The loss for images and texts
        loss_image = F.cross_entropy(logits_per_image, ground_truth)
        loss_text = F.cross_entropy(logits_per_text, ground_truth)

        # Return the average of the image and text loss
        return (loss_image + loss_text) / 2


def normalize_embeddings(*embeddings, p=2, dim=1):
    """
    Normalizes a list of embeddings using L2 normalization.

    Args:
        embeddings: List of torch tensors to be normalized.
        p: Norm order (default: 2 for L2 normalization).
        dim: Dimension along which to normalize (default: 1).

    Returns:
        List of normalized embeddings.
    """
    return [F.normalize(embed, p=p, dim=dim) for embed in embeddings]

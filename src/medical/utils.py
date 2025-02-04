import numpy as np
from typing import List, Optional, Dict,Union
from sklearn import metrics
import wandb
import torch
from src.shared_utils import *
from src.medical.dataset import create_datasets

def load_medical_data(train_dict, val_dict, test_dict, tokenizer, language_model, cfg):
    """
    Loads and initializes medical datasets based on the provided configuration.

    Args:
        language_model (SentenceTransformer, optional): The language model used for text processing.
        cfg (DictConfig): Configuration settings 

    Returns:
        tuple: Contains the train, validation, and test datasets, which are initialized based on 
               the configuration and processed using the language model.
    """
    return create_datasets(cfg, train_dict, val_dict, test_dict, tokenizer, language_model)


  
def get_metrics(
    preds: List[int],
    labels: List[int],
    probs: Optional[np.array] = None,
    multi_class: str = "ovr",
    class_names: Optional[List[str]] = None,
    use_wandb: bool = False,
) -> Dict:
    """
    Compute evaluation metrics including kappa, AUC, and confusion matrix.
    
    Args:
        preds (List[int]): Predicted class labels.
        labels (List[int]): True class labels.
        probs (Optional[np.array], optional): Predicted probabilities for each class. Defaults to None.
        multi_class (str, optional): Strategy for multiclass AUC calculation. Defaults to "ovr".
        class_names (Optional[List[str]], optional): List of class names. Defaults to None.
        use_wandb (bool, optional): Whether to log metrics to wandb. Defaults to False.
    
    Returns:
        Dict: Dictionary containing kappa score, confusion matrix, and AUC (if available).
    """
    labels = np.asarray(labels)
    
    # Compute Quadratic Weighted Kappa
    quadratic_weighted_kappa = metrics.cohen_kappa_score(labels, preds, weights="quadratic")
    
    # Generate Confusion Matrix
    cm = plot_confusion_matrix(
        labels.tolist(),
        preds,
        show_pct=True,
        cbar=False,
        names=class_names,
        normalize="true",
        title="Confusion Matrix",
    )
    
    metrics_dict = {"kappa": quadratic_weighted_kappa, "cm": cm}
    
    # Log Confusion Matrix to wandb if enabled
    if use_wandb:
        wandb_cm = wandb.plot.confusion_matrix(
            y_true=labels.tolist(),
            preds=preds,
            class_names=class_names,
        )
        metrics_dict["wandb_cm"] = wandb_cm
    
    # Compute AUC if probabilities are provided
    if probs is not None:
        auc = metrics.roc_auc_score(labels, probs, multi_class=multi_class)
        metrics_dict["auc"] = auc
    
    return metrics_dict

def get_preds_from_regression_logits(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Converts regression logits into discrete class predictions.
    
    Args:
        logits (torch.Tensor): The regression logits output from a model.
        num_classes (int): The total number of classes.
    
    Returns:
        torch.Tensor: Predicted class indices.
    """
    device = logits.device
    logits = logits.squeeze()  # Ensures correct tensor shape
    
    num_classes_tensor = torch.tensor([num_classes - 1], device=device)
    zero_tensor = torch.tensor([0], device=device)
    
    pred = torch.max(
        torch.min(torch.round(logits), num_classes_tensor),
        zero_tensor,
    )
    
    return pred

def plot_confusion_matrix(
    y_true: Union[List[float], np.array],
    y_pred: Union[List[float], np.array],
    show_pct: bool = False,
    cbar: bool = False,
    names: Optional[str] = None,
    normalize: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 300,
):
    """
    Computes & plots confusion matrix

    Args:
        y_true (_type_): array-like of shape (n_samples,)
            Ground truth (correct) target values.
        y_pred (_type_): array-like of shape (n_samples,)
            Estimated targets as returned by a classifier.
        show_pct (bool, optional): _description_. Defaults to False.
        cbar (bool, optional): _description_. Defaults to False.
        names (Optional[str], optional): _description_. Defaults to None.
        normalize (Optional[str], optional): _description_. Defaults to None.
        title (Optional[str], optional): _description_. Defaults to None.
        save_path (Optional[str], optional): _description_. Defaults to None.
        dpi (int, optional): _description_. Defaults to 150.
    """

    cm = metrics.confusion_matrix(y_true, y_pred, normalize=normalize)
    cm_unnorm = metrics.confusion_matrix(y_true, y_pred, normalize=None)

    if not normalize:
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        annot2 = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i][0]
                    if show_pct:
                        annot[i, j] = f"{c}/{s}"
                        annot2[i, j] = f"\n\n{p:.2f}%"
                    else:
                        annot[i, j] = f"{c}/{s}"
                elif c == 0:
                    annot[i, j] = f"{c}"
                    annot2[i, j] = ""
                else:
                    if show_pct:
                        annot[i, j] = f"{c}"
                        annot2[i, j] = f"\n\n{p:.2f}%"
                    else:
                        annot[i, j] = f"{c}"

    else:
        cm_sum = np.sum(cm_unnorm, axis=1, keepdims=True)
        annot = np.empty_like(cm).astype(str)
        annot2 = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm_unnorm[i, j]
                p = cm[i, j] * 100
                if i == j:
                    s = cm_sum[i][0]
                    if show_pct:
                        annot[i, j] = f"{c}/{s}"
                        annot2[i, j] = f"\n\n{p:.1f}%"
                    else:
                        annot[i, j] = f"{c}/{s}"
                elif c == 0:
                    annot[i, j] = f"{c}"
                    annot2[i, j] = ""
                else:
                    if show_pct:
                        annot[i, j] = f"{c}"
                        annot2[i, j] = f"\n\n{p:.1f}%"
                    else:
                        annot[i, j] = f"{c}"

    if names and len(names) == cm.shape[0]:
        labels = [f"{str(n).upper()}" for n in names]
    else:
        labels = [f"{i}" for i in range(cm.shape[0])]

    cm = pd.DataFrame(cm, index=labels, columns=labels)
    fig, ax = plt.subplots(dpi=dpi)

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        ax=ax,
        cmap="Blues",
        cbar=cbar,
        annot_kws={"size": "small"},
    )

    # Create a colormap with fully transparent colors
    cmap = sns.color_palette("Blues", as_cmap=True)
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[:, -1] = 0.0
    transparent_cmap = matplotlib.colors.ListedColormap(cmap_colors)
    sns.heatmap(
        cm,
        annot=annot2,
        fmt="",
        cmap=transparent_cmap,
        cbar=False,
        annot_kws={"size": "xx-small"},
    )

    ax.set_xlabel("Predicted", labelpad=10)
    ax.set_ylabel("Groundtruth", labelpad=10)
    if title is not None:
        ax.set_title(title, pad=10)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    return fig



def retrieval_kappa(cosine_sim_matrix, caption_column='microscopy', phase='test', config=None):
    """
    Computes the quadratic weighted kappa score for retrieval evaluation.
    
    Args:
        cosine_sim_matrix (torch.Tensor): Cosine similarity matrix between embeddings.
        caption_column (str, optional): Column to retrieve captions from. Defaults to 'microscopy'.
        phase (str, optional): Dataset phase ('test' or 'validation'). Defaults to 'test'.
        config (dict, optional): Configuration settings containing dataset paths. Defaults to None.
    
    Returns:
        tuple: Quadratic weighted kappa score, retrieval status, original captions,
               retrieved captions, and image IDs.
    """
    retrieved_report_indices = []
    retrieved_labels = []
    retrieval_status = []
    retrieved_captions = []
    
    # Load dataset based on phase
    if phase == 'test':
        all_data_path = config.data.test_path
    elif phase == 'validation':
        all_data_path = config.data.val_path
    
    all_dict = load_json(all_data_path)
    image_ids = list(all_dict.keys())
    original_labels = [all_dict[image_id]['label'] for image_id in image_ids]
    
    # Iterate through similarity matrix to find top-1 retrieved reports
    for _, similarities in enumerate(cosine_sim_matrix):
        top_1_index = np.argmax(similarities.cpu().detach().numpy())  # Get index of highest similarity
        retrieved_report_indices.append(top_1_index)
        
        # Retrieve label and caption of the top-1 matched report
        retrieved_label = all_dict[image_ids[top_1_index]]['label']
        retrieved_labels.append(retrieved_label)
        retrieved_caption = all_dict[image_ids[top_1_index]][caption_column]
        retrieved_captions.append(retrieved_caption)
    
    # Compute quadratic weighted kappa score
    quadratic_weighted_kappa = metrics.cohen_kappa_score(
        original_labels, retrieved_labels, weights="quadratic"
    )
    
    original_captions = [all_dict[image_id][caption_column] for image_id in image_ids]
    retrieval_status = [original == retrieved for original, retrieved in zip(original_labels, retrieved_labels)]
    
    return quadratic_weighted_kappa, retrieval_status, original_captions, retrieved_captions, image_ids



def image_retrieval(cosine_sim_matrix,caption_column,phase='test', top_k_values = [1,5,10], config=None):
    """
    Compute the retrieval accuracy and count the number of intersections between retrieved and original captions.

    Args:
        cosine_sim_matrix (torch.Tensor): Cosine similarity matrix of shape (num_images, num_images).
        k (int): Number of top similar images to retrieve.

    Returns:
        overall_accuracy (float): Overall retrieval accuracy.
        total_intersections (int): Total number of intersections between retrieved and original captions.
        avg_intersections (float): Average number of intersections per image.
    """
    if phase =='test':
           all_data_path = config.data.test_path
    elif phase =='validation':
            all_data_path = config.data.val_path
    all_dict = load_json(all_data_path)
    image_ids = list(all_dict.keys())  # List of all image IDs
    original_labels = [all_dict[image_id]['label'] for image_id in image_ids]  # Original labels
    retrieved_report_indices = []
    retrieval_status = {k: [] for k in top_k_values}  # Track whether the correct report was in top-k for each k
    retrieved_captions = []

    for idx, similarities in enumerate(cosine_sim_matrix):
        original_report_index = idx 
        # top k captions retrieved for image ids
        retrieved_k_captions = [] 
        for k in top_k_values:
            top_k_indices =  torch.topk(similarities, k=k, largest=True).indices.cpu().numpy().tolist()
            retrieved_report_indices.append(top_k_indices) 
            if original_report_index in top_k_indices:
                retrieval_status[k].append(True)
            else:
                retrieval_status[k].append(False)
        # Retrieve the top-1 caption for analysis
        top_1_index = torch.argmax(similarities).item()
 
        retrieved_caption = all_dict[image_ids[top_1_index]][caption_column]
        retrieved_captions.append(retrieved_caption)
   
    # Compute top-k retrieval accuracy for each k
    top_k_accuracy = {k: sum(retrieval_status[k]) / len(image_ids) for k in top_k_values}
    # Return results including top-k accuracy
    original_captions = [all_dict[image_id][caption_column] for image_id in image_ids]
    
    return top_k_accuracy, retrieval_status, original_captions, retrieved_captions, image_ids
      
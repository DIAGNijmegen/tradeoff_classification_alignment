
import os
import torch
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from typing import List, Optional, Dict

from src.coco.dataset import create_datasets
from src.shared_utils import *


import os

import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torchmetrics.classification import MultilabelAveragePrecision

def load_data(language_model, cfg):
    """
    Initializes datasets using COCO annotations and JSON datasets.

    Args:
        language_model (SentenceTransformer): Language model for text processing.
        cfg (DictConfig): Configuration settings.

    Returns:
        tuple: Train, validation, and test datasets.
    """    
    data_dir = cfg.data.work_dir
    tokenizer = None  # Can be modified if using NLP models
    
    train_dict = load_json(cfg.data.train_path)
    val_dict = load_json(cfg.data.val_path)
    test_dict = (
        load_json(cfg.data.test_path) if os.path.exists(cfg.data.test_path) else None
    )

    img_dir_train = os.path.join(data_dir, "train2017")
    img_dir_val = os.path.join(data_dir, "val2017")

    coco_train_instances = COCO(
        os.path.join(data_dir, "annotations", "instances_train2017.json")
    )
    coco_val_instances = COCO(
        os.path.join(data_dir, "annotations", "instances_val2017.json")
    )

    return create_datasets(
        cfg,
        coco_train_instances,
        coco_val_instances,
        train_dict,
        val_dict,
        test_dict,
        img_dir_train,
        img_dir_val,
        tokenizer,
        language_model,
    )



def load_coco_annotations(data_dir: Path):
    """
    Load COCO annotation file paths for training and validation datasets.

    Args:
        data_dir (str): Path to the dataset directory.

    Returns:
        tuple: Lists containing file paths to instance and caption annotations.
    """
    ann_file_train_captions = os.path.join(data_dir, 'annotations', 'captions_train2017.json')
    ann_file_train_instances = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    ann_file_val_captions = os.path.join(data_dir, 'annotations', 'captions_val2017.json')
    ann_file_val_instances = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
    
    instances = [ann_file_train_instances, ann_file_val_instances]
    captions = [ann_file_train_captions, ann_file_val_captions]
    return instances, captions

def get_category_mappings(coco_instance):
    """
    Generate category mappings for COCO dataset.

    Args:
        coco_instance (COCO): COCO instance annotation object.

    Returns:
        tuple: Dictionary mapping original category IDs to new IDs and dictionary mapping new IDs to category names.
    """
    categories = coco_instance.loadCats(coco_instance.getCatIds())
    
    category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
    category_names = {idx: cat['name'] for idx, cat in enumerate(categories)}
    return category_mapping, category_names

def process_image_annotations(coco_instance, coco_caption, origin : str, category_mapping: dict, category_names: dict):
    """
    Process COCO image annotations, extracting labels and captions.

    Args:
        coco_instance (COCO): COCO instance annotation object.
        coco_caption (COCO): COCO caption annotation object.
        origin (str): Dataset type ('train' or 'validation').
        category_mapping (dict): Mapping of original category IDs to contiguous IDs.
        category_names (dict): Mapping of contiguous IDs to category names.

    Returns:
        tuple: Dictionaries mapping image IDs to labels/captions and labels to image IDs.
    """
    image_id_to_labels_captions = {}
    label_to_image_ids = {}
    all_data = {}
    
    for img_id in coco_instance.getImgIds():
        labels, captions = get_image_labels_and_captions(coco_instance, coco_caption, img_id, category_mapping)
        if labels and captions:
            mapped_labels = [category_mapping[label] for label in labels]
            category_names_mapped = [category_names[cat] for cat in mapped_labels]
            handle_multi_label_case(img_id, mapped_labels, captions, origin, category_mapping, category_names_mapped, image_id_to_labels_captions, label_to_image_ids)
    
    return image_id_to_labels_captions, label_to_image_ids, all_data

def get_image_labels_and_captions(coco_instance, coco_caption, img_id:int, category_mapping: dict):
    """
    Retrieve labels and captions for a given image ID in COCO dataset.

    Args:
        coco_instance (COCO): COCO instance annotation object.
        coco_caption (COCO): COCO caption annotation object.
        img_id (int): Image ID.
        category_mapping (dict): Mapping of original category IDs to contiguous IDs.

    Returns:
        tuple: Set of labels and list of captions.
    """
    ann_ids = coco_instance.getAnnIds(imgIds=img_id)
    anns = coco_instance.loadAnns(ann_ids)
    labels = {ann['category_id'] for ann in anns if ann['category_id'] in category_mapping}
 
    ann_ids = coco_caption.getAnnIds(imgIds=img_id)
    anns = coco_caption.loadAnns(ann_ids)
    captions = [ann['caption'] for ann in anns]
    
    return labels, captions

def handle_multi_label_case(img_id: int, mapped_labels: list, captions: list, origin: str, category_mapping: dict, category_names: list, image_id_to_labels_captions: dict, label_to_image_ids: dict):
    """
    Handle multi-label classification cases for images in COCO dataset.

    Args:
        img_id (int): Image ID.
        mapped_labels (list): Mapped category labels.
        captions (list): List of image captions.
        origin (str): Dataset type ('train' or 'validation').
        category_mapping (dict): Mapping of original category IDs to contiguous IDs.
        category_names (list): List of category names.
        image_id_to_labels_captions (dict): Dictionary mapping image IDs to label and caption data.
        label_to_image_ids (dict): Dictionary mapping labels to image IDs.
    """
    image_id_to_labels_captions[img_id] = {
        'labels': mapped_labels,
        'captions': captions,
        'category_names': category_names,
        'origin': origin
    }
    
    for mapped_label in mapped_labels:
        if mapped_label not in label_to_image_ids:
            label_to_image_ids[mapped_label] = []
        label_to_image_ids[mapped_label].append(img_id)

def preprocess_train_val_test_coco(data_dir: Path, save_dir: Path, additional_name: str =''):
    """
    Preprocess COCO dataset by extracting train, validation, and test data.

    Args:
        data_dir (str): Path to COCO dataset.
        save_dir (str): Path to save processed JSON files.
        include_testset (bool, optional): Whether to include test set. Defaults to True.
        additional_name (str, optional): Additional suffix for saved files. Defaults to ''.

    Returns:
        tuple: Dictionaries containing image-label mappings, label-image mappings, and all data.
    """
    instances, captions = load_coco_annotations(data_dir)
    coco_instance = COCO(instances[0])
    category_mapping, category_names = get_category_mappings(coco_instance)
    
    for instance, cap, name in zip(instances, captions, ['train', 'validation']):
        coco_instance = COCO(instance)
        coco_caption = COCO(cap)
    
        image_id_to_labels_captions, label_to_image_ids, all_data = process_image_annotations(
            coco_instance, coco_caption, name, category_mapping, category_names)
        save_json(image_id_to_labels_captions, save_dir, f'{name}{additional_name}.json')
    return image_id_to_labels_captions, label_to_image_ids, all_data


def get_metrics_coco(
    preds: List[int],
    labels: List[int],
    probs: Optional[np.ndarray] = None,
    multi_class: str = "ovr",
    class_names: Optional[List[str]] = None,
    use_wandb: bool = False,
) -> Dict[str, float]:
    """
    Compute accuracy, AUC, and mAP for COCO dataset.
    
    Args:
        preds (List[int]): Predicted class labels.
        labels (List[int]): True class labels.
        probs (Optional[np.ndarray], optional): Predicted probabilities for each class. Defaults to None.
        multi_class (str, optional): Strategy for multiclass AUC calculation. Defaults to "ovr".
        class_names (Optional[List[str]], optional): List of class names. Defaults to None.
        use_wandb (bool, optional): Whether to log metrics to wandb. Defaults to False.
    
    Returns:
        Dict[str, float]: Dictionary containing precision, recall, f1-score, accuracy, AUC, and mAP.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    
    metrics_dict = {}
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    
    # Compute precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    metrics_dict.update({
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (labels == preds).mean()
    })
    
    if probs is not None:
        assert labels.shape == probs.shape, "Labels and probs must have the same shape"
        
        # Compute AUC for multi-label classification
        metrics_dict["auc"] = roc_auc_score(labels, probs, average="samples")
        
        # Convert numpy arrays to torch tensors
        labels = torch.from_numpy(labels).long()
        probs = torch.from_numpy(probs)
        
        # Compute mAP (COCO-style macro averaging)
        average_precision = MultilabelAveragePrecision(num_labels=labels.shape[1], average="macro")
        metrics_dict["mAP"] = average_precision(probs, labels).item()
    
    return metrics_dict


def coco_retrieval(cosine_sim_matrix,original_captions, k=10, phase= 'test', config=None):
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
   
    image_ids_test = list(all_dict.keys())
    top_n_retrieval_accuracy = []  # Store retrieval accuracy for each image
    total_intersections = 0        # Total number of intersections for all images
    
    all_retrieved_captions,all_original_captions = [], []
    all_accuracies = []
 
    for idx, similarities in enumerate(cosine_sim_matrix):
        # top k indices for original image at index idx
        top_n_indices = similarities.topk(k).indices.tolist()
        # top k captions retrieved for image ids
        retrieved_k_captions = [] 
        for retrieved_idx in top_n_indices:
                retrieved_k_captions.append(original_captions[retrieved_idx]) #5*k so not only retrieved but also other ones
        
        all_retrieved_captions.append(retrieved_k_captions)
        
        # 5 captions for the original image_id
        original_key = image_ids_test[idx]
      
        old_original_captions = all_dict[original_key]['captions'] 
        all_original_captions.append(old_original_captions)
       
        # Count the number of intersections
        intersections = sum(caption in old_original_captions for caption in retrieved_k_captions)
        total_intersections += intersections
          
        # if one of the retrieved captions is equal to one of the 5 original captions, then it's correct
        correct_retrieval = intersections > 0
        top_n_retrieval_accuracy.append(correct_retrieval)
        #per image whether it was correct retrieved or not
        all_accuracies.append(correct_retrieval)
    overall_accuracy = sum(top_n_retrieval_accuracy) / len(top_n_retrieval_accuracy)
    avg_intersections = total_intersections / len(top_n_retrieval_accuracy)
    return overall_accuracy, all_accuracies,all_retrieved_captions, all_original_captions


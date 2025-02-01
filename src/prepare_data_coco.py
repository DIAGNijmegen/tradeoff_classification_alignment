import os
import json
import random
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from sklearn.model_selection import KFold, train_test_split

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

def get_category_mappings(coco_instance, exclude_category: list=[]):
    """
    Generate category mappings for COCO dataset.

    Args:
        coco_instance (COCO): COCO instance annotation object.
        exclude_category (list): List of categories to exclude.

    Returns:
        tuple: Dictionary mapping original category IDs to new IDs and dictionary mapping new IDs to category names.
    """
    categories = coco_instance.loadCats(coco_instance.getCatIds())
    categories = [cat for cat in categories if cat['name'] not in exclude_category]
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

def preprocess_train_val_test_coco(data_dir: Path, save_dir: Path, exclude_category: list=[], include_testset: bool=True, additional_name: list =''):
    """
    Preprocess COCO dataset by extracting train, validation, and test data.

    Args:
        data_dir (str): Path to COCO dataset.
        save_dir (str): Path to save processed JSON files.
        exclude_category (list, optional): Categories to exclude. Defaults to [].
        include_testset (bool, optional): Whether to include test set. Defaults to True.
        additional_name (str, optional): Additional suffix for saved files. Defaults to ''.

    Returns:
        tuple: Dictionaries containing image-label mappings, label-image mappings, and all data.
    """
    instances, captions = load_coco_annotations(data_dir)
    coco_instance = COCO(instances[0])
    category_mapping, category_names = get_category_mappings(coco_instance, exclude_category)
    
    for instance, cap, name in zip(instances, captions, ['train', 'validation']):
        coco_instance = COCO(instance)
        coco_caption = COCO(cap)
    
        image_id_to_labels_captions, label_to_image_ids, all_data = process_image_annotations(
            coco_instance, coco_caption, name, category_mapping, category_names)
        save_json(image_id_to_labels_captions, save_dir, f'{name}{additional_name}.json')
    return image_id_to_labels_captions, label_to_image_ids, all_data

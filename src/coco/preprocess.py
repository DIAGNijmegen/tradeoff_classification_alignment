import os
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split, KFold

from src.coco.preprocess import *


def load_coco_annotation_paths(data_dir: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """Load paths to COCO annotation files."""
    ann_dir = Path(data_dir) / 'annotations'
    instances = [ann_dir / 'instances_train2017.json', ann_dir / 'instances_val2017.json']
    captions = [ann_dir / 'captions_train2017.json', ann_dir / 'captions_val2017.json']
    return list(map(str, instances)), list(map(str, captions))


def get_category_mappings(coco: COCO, exclude: List[str]) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Generate mappings from category IDs to indices and names."""
    categories = [cat for cat in coco.loadCats(coco.getCatIds()) if cat['name'] not in exclude]
    id_to_idx = {cat['id']: idx for idx, cat in enumerate(categories)}
    idx_to_name = {idx: cat['name'] for idx, cat in enumerate(categories)}
    return id_to_idx, idx_to_name


def get_labels_and_captions(coco_ins: COCO, coco_cap: COCO, img_id: int, cat_map: Dict[int, int]) -> Tuple[List[int], List[str]]:
    labels = {ann['category_id'] for ann in coco_ins.loadAnns(coco_ins.getAnnIds(imgIds=img_id)) if ann['category_id'] in cat_map}
    captions = [ann['caption'] for ann in coco_cap.loadAnns(coco_cap.getAnnIds(imgIds=img_id))]
    return list(labels), captions


def collect_image_annotations(
    coco_ins: COCO,
    coco_cap: COCO,
    dataset: str,
    cat_map: Dict[int, int],
    cat_names: Dict[int, str]
) -> Tuple[Dict[int, dict], Dict[int, List[int]]]:
    """Extract labels and captions for all images using COCO API."""
    img_to_data, label_to_imgs = {}, {}
    for img_id in coco_ins.getImgIds():
        labels, captions = get_labels_and_captions(coco_ins, coco_cap, img_id, cat_map)
        if labels and captions:
            mapped_labels = [cat_map[lbl] for lbl in labels]
            names = [cat_names[lbl] for lbl in mapped_labels]
            img_to_data[img_id] = {'labels': mapped_labels, 'captions': captions, 'category_names': names, 'origin': dataset}
            for lbl in mapped_labels:
                label_to_imgs.setdefault(lbl, []).append(img_id)
    return img_to_data, label_to_imgs


def save_json(data: Union[dict, DictConfig], path: Union[str, Path], name: str = None) -> None:
    if isinstance(data, DictConfig):
        data = OmegaConf.to_container(data, resolve=True)
    out_path = Path(path) / name if name else Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def preprocess_coco_data(data_dir: str, save_dir: str, exclude: List[str]) -> Dict[int, dict]:
    """Preprocess COCO annotations and save train/validation data to JSON."""
    ins_paths, cap_paths = load_coco_annotation_paths(data_dir)
    coco = COCO(ins_paths[0])
    cat_map, cat_names = get_category_mappings(coco, exclude)
    all_data = {}

    for ins, cap, split in zip(ins_paths, cap_paths, ['train', 'validation']):
        coco_ins, coco_cap = COCO(ins), COCO(cap)
        img_data, _, = collect_image_annotations(coco_ins, coco_cap, split, cat_map, cat_names)
        all_data.update(img_data)
        save_json(img_data, save_dir, f'{split}.json')
    return all_data


def split_train_val_test(data: Dict, ratios=(0.7, 0.15, 0.15), seed=42) -> Tuple[dict, dict, dict]:
    """Split task-specific finetuning data into train, validation, and test sets."""
    items = list(data.items())
    train, temp = train_test_split(items, test_size=1 - ratios[0], random_state=seed)
    val, test = train_test_split(temp, test_size=ratios[2] / (ratios[1] + ratios[2]), random_state=seed)
    return dict(train), dict(val), dict(test)


def k_fold_split(data: dict, k: int, save_dir: str, seed=42) -> None:
    """Split data into k train/val folds and save them to disk."""
    samples = np.array(list(data.items()), dtype=object)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_path = Path(save_dir) / 'folds'
    fold_path.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(samples)):
        train = {k: v for k, v in samples[train_idx]}
        val = {k: v for k, v in samples[val_idx]}
        save_json(train, fold_path, f'train_fold{fold}.json')
        save_json(val, fold_path, f'val_fold{fold}.json')


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    
    """
    This script preprocesses the COCO images2017 and val2017 sets into two seperate  datasets.

    1. Vision Encoder Fine-Tuning Data (Supervised):
    - 70% of the available images are reserved for training a multi-label classification model to obtain a well-performing task-specific vision model. Specifically, we finetune the ViT-Base architecture, "google/vit-base-patch16-224" (Wu et al., 2020) for multi-label classification on the 80 classes in the dataset.
    - These are split into training/validation/test sets for task-specific vision encoder optimization.
    - This dataset is stored under /data/vision_encoder_finetuning/

    2. Contrastive Tuning Image-Text Data:
    - The remaining 30% of the images (and corresponding captions) are used for the contrastive tuning experiments.
    - We create 5 folds using K-Fold cross-validation and an independent testset 
    - This dataset is stored under /data/contrastive_learning/
    """

    image_dir = '/path/to/image/directory'  # Replace with the actual path to the COCO images directory
    base_dir = './data/'
    seed = 42
    save_dir_contrastive_tuning = Path(base_dir) / 'contrastive_tuning'
    save_dir_vison = Path(base_dir) / 'vision_encoder_finetuning'

    data = preprocess_coco_data(image_dir, save_dir_contrastive_tuning, exclude=[])
    train_data = load_json(save_dir_contrastive_tuning / 'train.json')
    val_data = load_json(save_dir_contrastive_tuning / 'validation.json')

    save_json(val_data, save_dir_contrastive_tuning, 'testset.json')  # use full val as test

    train_items = list(train_data.items())
    train_items = np.array(train_items, dtype=object)
    train_cv, vision_only = train_test_split(train_items, test_size=0.8, random_state=seed)

    k_fold_split(dict(train_cv), k=5, save_dir=save_dir_contrastive_tuning, seed=seed)
    os.remove(save_dir_contrastive_tuning / 'train.json')
    os.remove(save_dir_contrastive_tuning / 'validation.json')

    vision_data = dict(vision_only)
    train, val, test = split_train_val_test(vision_data)
  
    save_json(train, save_dir_vison, 'train.json')
    save_json(val, save_dir_vison, 'val.json')
    save_json(test, save_dir_vison, 'test.json')
    print(f"Saved vision encoder finetuning sets in {save_dir_vison}")
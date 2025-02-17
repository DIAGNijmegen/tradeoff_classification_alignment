import torch
import random
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
from PIL import Image



class ImageTextCOCO(Dataset):
    """
    A PyTorch Dataset class for loading COCO image-text data with multi-label classification.

    Args:
        coco (COCO): COCO annotation object.
        dataset_dict (dict): Dictionary containing image IDs, labels, and captions.
        img_dir (str): Path to the image directory.
        tokenizer (callable, optional): Tokenizer function for text processing. Defaults to None.
        language_model (SentenceTransformer, optional): Language model for embeddings. Defaults to None.
        num_classes (int, optional): Number of classes for multi-label classification. Defaults to 80.
    """
    def __init__(self, coco, dataset_dict, img_dir, tokenizer=None, language_model=None, num_classes=80):
        self.coco = coco
        self.dataset_dict = dataset_dict
        self.image_ids = list(dataset_dict.keys())
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.num_classes = num_classes
        self.split = 'train' if 'train' in img_dir else 'val'
        
        self.labels = [dataset_dict[img_id]["labels"] for img_id in self.image_ids]
        self.captions = [dataset_dict[img_id]["captions"] for img_id in self.image_ids]
        
        self.transform = self._get_transforms()

    def _get_transforms(self):
        """
        Define image transformations based on dataset split (train/validation).
        
        Returns:
            transforms.Compose: Image transformations for preprocessing.
        """
        if self.split == "train":
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Image tensor, positive caption, negative caption, and label tensor.
        """
        img_id = int(self.image_ids[idx])
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info.get("file_name", img_info.get("image_name")))
        image = self.transform(Image.open(img_path).convert("RGB"))
        
        label_tensor = self._get_label_tensor(self.labels[idx])
        pos_caption, neg_caption = self._get_captions(idx)
        if isinstance(self.language_model, SentenceTransformer):
            return image, pos_caption, neg_caption, label_tensor
        
        return image, self._tokenize(pos_caption), self._tokenize(neg_caption), label_tensor

    def _get_label_tensor(self, label_list):
        """
        Convert label list into a multi-hot encoded tensor.

        Args:
            label_list (list): List of label indices.
        
        Returns:
            torch.Tensor: Multi-hot encoded label tensor.
        """
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float)
        label_tensor[label_list] = 1
        return label_tensor

    def _get_captions(self, idx):
        """
        Retrieve a positive caption from the same image and a negative caption from another image.

        Args:
            idx (int): Index of the sample.
        
        Returns:
            tuple: Positive caption and negative caption.
        """
        pos_caption = random.choice(self.captions[idx])
        all_captions = [caption for i, captions in enumerate(self.captions) if i != idx for caption in captions]
        neg_caption = random.choice(all_captions)
        return pos_caption, neg_caption

    def _tokenize(self, text):
        """
        Tokenize text using the provided tokenizer.

        Args:
            text (str): Input text to tokenize.
        
        Returns:
            dict: Tokenized output in PyTorch tensor format.
        """
        return self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)

    @classmethod
    def create_dataset(cls, coco, dataset_dict, img_dir, tokenizer, language_model, num_classes):
        return cls(coco, dataset_dict, img_dir, tokenizer, language_model, num_classes)



def create_datasets(cfg, coco_train_instances, coco_val_instances, train_dict, val_dict, test_dict, img_dir_train, img_dir_val, tokenizer, model_l):
    """
        Class method to create a dataset instance.
        
        Args:
            coco (COCO): COCO annotation object.
            dataset_dict (dict): Dictionary of dataset annotations.
            img_dir (str): Path to the image directory.
            tokenizer (callable): Tokenizer for text processing.
            language_model (SentenceTransformer): Language model for embedding.
            num_classes (int): Number of classification labels.
        
        Returns:
            ImageTextCOCO: An instance of the dataset class.
        """
    datasets = {
        "train": (coco_train_instances, train_dict, img_dir_train),
        "val": (coco_train_instances, val_dict, img_dir_train),
        "test": (coco_val_instances, test_dict, img_dir_val) if os.path.exists(cfg.data.test_path) else None
    }
    
    return [
        ImageTextCOCO.create_dataset(coco, d_dict, img_dir, tokenizer, model_l, cfg.num_classes)
        if coco else None
        for coco, d_dict, img_dir in datasets.values()
    ]
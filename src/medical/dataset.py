import torch
import random
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
from PIL import Image

class ImageTextDatasetMED(Dataset):
    """
    A PyTorch dataset for medical image-text data.
    
    Args:
        dataset_dict (dict): Dictionary containing image IDs, captions, and labels.
        language_model (SentenceTransformer, optional): Language model for embeddings. Defaults to None.
        tokenizer (callable, optional): Tokenizer function for text processing. Defaults to None.
        **config (dict): Additional configuration settings.
    """
    def __init__(self, dataset_dict, language_model=None, tokenizer=None, **config):
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.local_features_dir = config['vision']['local_features_dir']
        
        # Extract image IDs and captions from the dataset
        self.image_ids = list(dataset_dict.keys())
        self.caption_column = config['data']['caption_column']
        self.captions = [dataset_dict[img_id][self.caption_column] for img_id in self.image_ids]
        self.labels = [dataset_dict[img_id]['label'] for img_id in self.image_ids]
   
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.captions)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample.
        
        Returns:
            tuple: Feature path, tokenized positive report, tokenized negative report, and label tensor.
        """
        # Get the image feature path
        feature_path = os.path.join(self.local_features_dir, f"{self.image_ids[idx]}.pt")
        
        # Obtain the positive report and label
        positive_report = self.captions[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Select a negative report with a different label
        negative_report = self._get_negative_report(idx)
        
        if isinstance(self.language_model, SentenceTransformer):
            # Return raw text for SentenceTransformer (no tokenization required)
            return feature_path, positive_report, negative_report, label
        
        # Tokenize the positive and negative reports
        positive_report_tokenized = self._tokenize_report(positive_report)
        negative_report_tokenized = self._tokenize_report(negative_report)
        
        return feature_path, positive_report_tokenized, negative_report_tokenized, label

    def _get_negative_report(self, idx):
        """
        Selects a random negative report with a different label.

        Args:
            idx (int): Index of the current sample.
        
        Returns:
            str: A negative report from a different class.
        """
        negative_idx = random.choice(
            [i for i in range(len(self.captions)) if self.labels[i] != self.labels[idx]]
        )
        return self.captions[negative_idx]

    def _tokenize_report(self, report):
        """
        Tokenizes a report based on the tokenizer provided.

        Args:
            report (str): Input text to be tokenized.
        
        Returns:
            dict: Tokenized output in PyTorch tensor format.
        """
        return self.tokenizer.encode_plus(
            report,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

def create_datasets(cfg, train_dict, val_dict, test_dict, tokenizer, language_model):
    """
    Creates datasets for the Medical dataset.

    Args:
        cfg (DictConfig): Configuration settings.
        train_dict (dict): Dictionary for training data.
        val_dict (dict): Dictionary for validation data.
        test_dict (dict): Dictionary for test data.
        tokenizer (callable): Tokenizer for text processing.
        language_model (SentenceTransformer): Language model for embeddings.

    Returns:
        tuple: Train, validation, and test datasets.
    """
    datasets = {
        "train": train_dict,
        "val": val_dict,
        "test": test_dict if test_dict else None
    }
    
    return [
        ImageTextDatasetMED(
            dataset_dict=d,
            language_model=language_model,
            tokenizer=tokenizer,
            vision=cfg.vision,
            data=cfg.data,
        ) if d else None
        for d in datasets.values()
    ]

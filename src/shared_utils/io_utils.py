import os
import json
import glob
from pathlib import Path
from typing import Any, Optional, Union,Tuple,Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from omegaconf import DictConfig, OmegaConf

def save_json(data, save_path_or_dir, name=None):
    """
    Save data to a JSON file. Automatically handles OmegaConf DictConfig,
    ensures the parent directory exists, and supports either a full path or a directory + filename.

    Args:
        data (dict or DictConfig): The data to save.
        save_path_or_dir (str or Path): Either full path to JSON file, or directory to save in.
        name (str, optional): If provided, treated as the filename to save within the given directory.
    """
    # Convert OmegaConf to plain dict
    if isinstance(data, DictConfig):
        data = OmegaConf.to_container(data, resolve=True)

    # Resolve final path
    if name is None:
        output_path = Path(save_path_or_dir)
    else:
        output_path = Path(save_path_or_dir) / name

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


 
def load_json(file_path: Union[str, Path]) -> Any:
    """Loads and returns the contents of a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)



def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    loss: float,
    checkpoint_dir: Union[str, os.PathLike]
) -> str:
    """
    Saves the model checkpoint to a specified directory.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): Vision model to save.
        optimizer (Optimizer): Optimizer whose state will be saved.
        loss (float): Training loss value.
        checkpoint_dir (str or Path): Directory to save the checkpoint.

    Returns:
        str: Path to the saved checkpoint file.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path




def clean_previous_checkpoint(previous_checkpoint: Optional[str], new_checkpoint: str) -> str:
    """
    Deletes the previous checkpoint if it exists to save space and avoid confusion.

    Args:
        previous_checkpoint (str or None): Path to the previous checkpoint file.
        new_checkpoint (str): Path to the new checkpoint file.

    Returns:
        str: Updated checkpoint path.
    """
    if previous_checkpoint and os.path.exists(previous_checkpoint):
        os.remove(previous_checkpoint)
        print(f"Deleted old checkpoint: {previous_checkpoint}")
    return new_checkpoint

# Function to find the latest checkpoint
def get_latest_checkpoint(checkpoint_dir: Union[str, os.PathLike]) -> Optional[str]:
    """
    Finds the most recently modified checkpoint file in the directory.

    Args:
        checkpoint_dir (str or Path): Path to directory containing checkpoints.

    Returns:
        str or None: Path to the latest checkpoint, or None if not found.
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint



def load_checkpoint(
    model_v: nn.Module,
    optim_vision: Optional[optim.Optimizer],
    checkpoint_dir: Path
) -> Tuple[int, Optional[Path]]:
    """
    Loads the latest checkpoint to resume training if available.

    Args:
        model_v (torch.nn.Module): Vision model.
        optim_vision (torch.optim.Optimizer): Optimizer for the vision model.
        checkpoint_dir (Path): Path to the checkpoint directory.

    Returns:
        tuple: Starting epoch number and latest checkpoint path.
    """
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    start_epoch = 0

    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint)
        model_v.load_state_dict(checkpoint["model_state_dict"])
        if optim_vision:
            optim_vision.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(
            f"Resumed from checkpoint: {latest_checkpoint} (Starting at epoch {start_epoch})"
        )
    else:
        print("No checkpoint found. Starting from scratch.")

    return start_epoch, latest_checkpoint




def save_results_dict(
    train_dict: Optional[Dict[str, Any]] = None,
    validation_dict: Optional[Dict[str, Any]] = None,
    validation_testset_dict: Optional[Dict[str, Any]] = None,
    test_dict: Optional[Dict[str, Any]] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> None:
 
    if save_dir is not None:
        # Save the results dictionary to a JSON file
        with open(Path(save_dir, 'trainer_state.json'), 'w') as f:
            json.dump(train_dict, f, indent=4)
            
        with open(Path(save_dir,'validation_state.json'), 'w') as f:
            json.dump(validation_dict, f, indent=4)
            
        with open(Path(save_dir,'best_valid_model_on_testset.json'), 'w') as f:
            json.dump(validation_testset_dict, f, indent=4)
        
        with open(Path(save_dir,'test_results.json'), 'w') as f:
            json.dump(test_dict, f, indent=4)




def setup_directories(base_dir: Path) -> Tuple[Path, Path]:
    """
    Creates necessary output directories for storing checkpoints and results.

    Args:
        base_dir (Path): Base directory path.

    Returns:
        tuple: Paths to checkpoint and result directories.
    """
    checkpoint_dir = Path(base_dir, "checkpoints")
    result_dir = Path(base_dir, "results")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, result_dir

            
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf


def log_results(
    results: Dict[str, Any],
    phase: str,
    log_dict: Dict[str, Dict[str, Any]],
    log_to_wandb: Dict[str, List[str]]
) -> None:
    """
    Logs and saves results to wandb.

    Args:
        results (dict): Results dictionary.
        phase (str): Training phase ('train', 'validation', 'test').
        log_dict (dict): Logging dictionary.
        log_to_wandb (dict): Dictionary for W&B logging.
    """
    results.pop("cm", None)
    results.pop("wandb_cm", None)

    update_log_dict(
        phase,
        results,
        log_dict,
        to_log=[e for e in log_to_wandb[phase] if "cm" not in e],
        step="epoch",
    )
def plot_cm(
    results: dict,
    log_to_wandb: dict,
    dir: Optional[Path] = None,
    phase: Optional[str] = "test",
    wandb_enable: bool = False,
):

    for r, v in results.items():
        if isinstance(v, float):
            v = round(v, 5)
        if r == "cm":
            save_path = Path(dir, f"{phase}_cm.png")
            v.savefig(save_path, bbox_inches="tight")
            plt.close(v)
        if wandb_enable and r in log_to_wandb[phase]:
            if r == "cm":
                wandb.log({f"{phase}/{r}": wandb.Image(str(save_path))})
            else:
                wandb.log({f"{phase}/{r}": v})
        elif "cm" not in r:
            print(f"{phase} {r}: {v}")


def prepare_image_for_wandb(image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Prepares an image for logging to Weights & Biases (W&B) by converting
    tensors to NumPy, handling channel formats, and normalizing.

    Args:
        image (Tensor or ndarray): The image to format.

    Returns:
        np.ndarray: Processed image in HWC format with 3 channels (RGB).
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    if image.ndim == 3 and image.shape[0] in [1, 3]:  # CHW
        image = np.transpose(image, (1, 2, 0))  # to HWC

    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    if image.ndim == 2:  # Grayscale
        image = np.stack((image,) * 3, axis=-1)
    elif image.shape[2] == 1:  # Single-channel image
        image = np.concatenate([image] * 3, axis=2)

    return image



def initialize_wandb(
    cfg: DictConfig,
    key: Optional[str] = "",
) -> wandb.sdk.wandb_run.Run:
    """_summary_

    Args:
        cfg (DictConfig): _description_
        key (Optional[str], optional): _description_. Defaults to "".

    Returns:
        _type_: wanb run
    """
    command = f"wandb login {key}"
    subprocess.call(command, shell=True)
    if cfg.wandb.tags == None:
        tags = []
    else:
        tags = [str(t) for t in cfg.wandb.tags]
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if cfg.resume_id:
        print(f"Resuming wandb run with run_id: {cfg.resume_id}")
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.username,
            #name=cfg.wandb.exp_name,
            name= f'l={cfg.lambda_param}_{cfg.resume_id}',
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            config=config,
            tags=tags,
            id=cfg.resume_id, # resuming specific run 
            resume="must",
        )
    
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.username,
            name=cfg.wandb.exp_name,
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            config=config,
            tags=tags,
        )
    config_file_path = Path(run.dir, "run_config.yaml")
    d = OmegaConf.to_container(cfg, resolve=True)
    with open(config_file_path, "w+") as f:
        write_dictconfig(d, f)
        wandb.save(str(config_file_path))
        f.close()
    return run


def write_dictconfig(d, f, child: bool = False, ntab=0):
    """
    Recursively writes a nested dictionary (e.g., from OmegaConf.to_container)
    to a YAML-like format with indentation.

    Args:
        d (dict): The nested dictionary to write.
        f (TextIO): A writable file-like object (e.g., an open file).
        child (bool, optional): Whether this is a nested child block. Defaults to False.
        ntab (int, optional): Current indentation level (used internally). Defaults to 0.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            if not child:
                f.write(f"{k}:\n")
            else:
                for _ in range(ntab):
                    f.write("\t")
                f.write(f"- {k}:\n")
            write_dictconfig(v, f, True, ntab=ntab + 1)
        else:
            if isinstance(v, list):
                if not child:
                    f.write(f"{k}:\n")
                    for e in v:
                        f.write(f"\t- {e}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"{k}:\n")
                    for e in v:
                        for _ in range(ntab):
                            f.write("\t")
                        f.write(f"\t- {e}\n")
            else:
                if not child:
                    f.write(f"{k}: {v}\n")
                else:
                    for _ in range(ntab):
                        f.write("\t")
                    f.write(f"- {k}: {v}\n")


def update_log_dict(
    prefix: str,
    results: Dict[str, Any],
    log_dict: Dict[str, Any],
    step: Optional[str] = "step",
    to_log: Optional[List[str]] = None,
) -> None:
    """_summary_

    Args:
        prefix (_type_): _description_
        results (_type_): _description_
        log_dict (_type_): _description_
        step (Optional[str], optional): _description_. Defaults to "step".
        to_log (Optional[List[str]], optional): _description_. Defaults to None.
    """
    if not to_log:
        to_log = list(results.keys())
    for r, v in results.items():
        if r in to_log:
            wandb.define_metric(f"{prefix}/{r}", step_metric=step)
            log_dict.update({f"{prefix}/{r}": v})


def compute_time(start_time, end_time):
    """
        Computes the elapsed time in minutes and seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


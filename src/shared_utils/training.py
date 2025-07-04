from typing import List, Optional,Dict,Optional, Tuple, Any
import time

from pathlib import Path
import numpy as np
import torch.nn as nn
import torch
import importlib
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import DictConfig
from torch.optim import Optimizer
from transformers import PreTrainedTokenizer, AutoTokenizer


from shared_utils.io_utils import save_json, clean_previous_checkpoint
from shared_utils.logging_utils import log_results, plot_cm
from shared_utils.factories import ModelFactory, LossFactory, OptimizerFactory
from shared_utils.early_stopping import EarlyStopping


def load_models(cfg: DictConfig, device: str) -> Tuple[nn.Module, nn.Module, Optional[PreTrainedTokenizer]]:
    """
    Loads and prepares vision and language models.

    Args:
        cfg (DictConfig): Configuration settings.
        device (str): Device to load models onto (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: Vision model, language model, and tokenizer (if applicable).
    """
    model_v = ModelFactory(cfg.vision.model_type, cfg).get_model()
    print(f"Successfully loaded Vision Model: {cfg.vision.model_type}")

    model_l = ModelFactory(cfg.language.model_type, cfg).get_model()
    for param in model_l.parameters():
        param.requires_grad = False  # Freeze language model
    print(f"Successfully loaded Language Model: {cfg.language.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.language.base_dir, local_files_only=True) \
        if cfg.language.base_dir is not None else None

    return model_v.to(device), model_l.to(device), tokenizer


def load_training_components(cfg: DictConfig, model_v: nn.Module) -> Tuple[nn.Module, nn.Module, Optional[optim.Optimizer]]:
    """
    Loads classification & contrastive loss functions and optimizer.

    Args:
        cfg (DictConfig): Configuration settings.
        model_v (torch.nn.Module): Vision model.

    Returns:
        tuple: Classification criterion, contrastive criterion, and optimizer for vision model.
    """
    classification_criterion = LossFactory(cfg.training.classification_loss).get_loss()
    contrastive_criterion = LossFactory(
        cfg.training.contrastive_loss, margin=cfg.training.loss_options.margin
    ).get_loss()

    optim_vision = None
    if cfg.optimizers.optim_vision.enable:
        vision_model_params = filter(lambda p: p.requires_grad, model_v.parameters())
        optim_vision = OptimizerFactory(
            cfg.optimizers.optim_vision.name,
            vision_model_params,
            lr=cfg.optimizers.optim_vision.lr,
            weight_decay=cfg.optimizers.optim_vision.wd,
        ).get_optimizer()
        print("Optimizer for vision model initialized")

    return classification_criterion, contrastive_criterion, optim_vision



def run_training(
    cfg: Any,
    trainset: Any,
    validationset: Any,
    testset: Any,
    model_v: nn.Module,
    model_l: nn.Module,
    optim_vision: Optional[Optimizer],
    classification_criterion: nn.Module,
    contrastive_criterion: nn.Module,
    checkpoint_dir: Path,
    result_dir: Path,
    log_to_wandb: Dict[str, List[str]],
) -> Tuple[nn.Module, Dict[str, Optional[Dict[str, Any]]], Dict[str, Optional[Dict[str, Any]]]]:
    """
    Handles training, validation, and logging.
    """
    # Construct module name dynamically
    module_name = f"src.{cfg.experiment_name}.train"
    train_module = importlib.import_module(module_name)
    # Import everything from the module into the global namespace
    globals().update({name: getattr(train_module, name) for name in dir(train_module) if not name.startswith("_")})

    # save results as dict
    final_training_results = {
        str(epoch): None for epoch in np.arange(cfg.training.num_epochs)
    }
    final_validation_results = {
        str(epoch): None for epoch in np.arange(cfg.training.num_epochs)
    }

    early_stopping = EarlyStopping(
        cfg.early_stopping.metric,
        cfg.early_stopping.mode,
        cfg.early_stopping.patience,
        cfg.early_stopping.min_epoch,
        checkpoint_dir=checkpoint_dir,
        save_all=cfg.early_stopping.save_all,
    )
    previous_checkpoint = None
    stop = False

    with tqdm.tqdm(
        range(cfg.training.num_epochs),
        desc="Contrastive Tuning",
        unit="epoch",
        ncols=100,
        leave=False,
    ) as t:
        for epoch in t:
            epoch_start_time = time.time()
            log_dict = {"epoch": epoch} if cfg.wandb.enable else {}

            if epoch == 0:
                print("Computing baseline results on validation set prior to training")
                # baseline validationset
                baseline_validation = tune_epoch(
                    epoch=epoch,
                    dataset=validationset,
                    vision_model=model_v,
                    language_model=model_l,
                    classification_criterion= classification_criterion,
                    contrastive_criterion=contrastive_criterion,
                    cfg=cfg,
                )
                if cfg.wandb.enable:
                    log_results(
                         baseline_validation, "validation", log_dict, log_to_wandb
                    ) 
                final_validation_results[str(epoch)] = baseline_validation
                save_json(
                    baseline_validation, result_dir, name="baseline_validation.json"
                )
                print("Computing baseline results on testset prior to training")
                # baseline testset
                baseline_test = tune_epoch(
                    epoch=epoch,
                    dataset=testset,
                    vision_model=model_v,
                    language_model=model_l,
                    classification_criterion= classification_criterion,
                    contrastive_criterion=contrastive_criterion,
                    phase='test',
                    cfg=cfg,
                )
               
                if cfg.wandb.enable:
                    log_results( baseline_test, "test", log_dict, log_to_wandb) 
                save_json(baseline_validation, result_dir, name="baseline_testset.json")

                # logging
                if cfg.wandb.enable:
                    wandb.log(log_dict, step=epoch)

            else:
                # train epoch
                train_results, checkpoint = train_epoch(
                    epoch=epoch,
                    dataset=trainset,
                    vision_model=model_v,
                    language_model=model_l,
                    optimizer_vision=optim_vision,
                    classification_criterion=classification_criterion,
                    contrastive_criterion=contrastive_criterion,
                    checkpoint_dir=checkpoint_dir,
                    cfg=cfg)
                previous_checkpoint = clean_previous_checkpoint(
                    previous_checkpoint, checkpoint
                )
                final_training_results[str(epoch)] = train_results
                if cfg.wandb.enable:
                    log_results(
                     train_results, "train", log_dict, log_to_wandb
                ) 

                # validation epoch
                validation_results = tune_epoch(
                    epoch=epoch,
                    dataset=validationset,
                    vision_model=model_v,
                    language_model=model_l,
                    classification_criterion= classification_criterion,
                    contrastive_criterion=contrastive_criterion,
                    cfg=cfg)
                
                final_validation_results[str(epoch)] = validation_results
                if cfg.wandb.enable:
                    log_results(
                    validation_results, "validation", log_dict, log_to_wandb
                ) 

                early_stopping(epoch, model_v, validation_results)
                if early_stopping.early_stop and cfg.early_stopping.enable:
                    best_metrics = final_validation_results[
                        str(early_stopping.best_epoch)
                    ]
                    best_metrics["epoch"] = early_stopping.best_epoch
                    save_json(
                        best_metrics, result_dir, name="best_epoch_val_results_ES.json"
                    )
                    print(f"Stopping early after epoch: {epoch}")
                    stop = True

                if cfg.wandb.enable:
                    wandb.log(log_dict, step=epoch)

            print(
                f"End of epoch {epoch}/{cfg.training.num_epochs} \t Time Taken: {time.time() - epoch_start_time:.2f}s"
            )

            if stop:
                break

    return model_v, final_training_results, final_validation_results



def evaluate_final_model(
    cfg: Any,
    model_v: nn.Module,
    model_l: nn.Module,
    testset: Any,
    log_to_wandb: Dict[str, List[str]],
    result_dir: Path,
) -> Dict[str, Any]:
    """
    Evaluates the final model on the test set.
    """
    test_results = test(
        testset,
        model_v,
        model_l,
        cfg
    )
    
    plot_cm(
        test_results, log_to_wandb, result_dir, "test", wandb_enable=cfg.wandb.enable
    )

    # Exclude specific keys (like 'cm' and 'wandb_cm')
    json_ready_results = {k: v for k, v in test_results.items() if k not in {'cm', 'wandb_cm'}}
    save_json(json_ready_results, result_dir, name="test_results.json")
   
    if cfg.wandb.enable:
        wandb.log({"test": json_ready_results}, step=cfg.training.num_epochs)

    return json_ready_results










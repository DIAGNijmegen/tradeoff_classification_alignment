import wandb
import time
import datetime
import tqdm
import os
import hydra
import torch
import numpy as np
from pathlib import Path
from omegaconf import  DictConfig
from src.coco.utils import *
from src.medical.utils import load_medical_data
from src.shared_utils.io_utils import (
    load_checkpoint,
    save_results_dict,
    setup_directories,
)
from src.shared_utils.logging_utils import (
    initialize_wandb,
)   
from src.shared_utils.training import (
    load_models,
    load_training_components,
    run_training,
    evaluate_final_model,
)
@hydra.main(
    version_base="1.2.0",
    config_path="../config",
    config_name="coco",
)
def main(cfg: DictConfig):
   
    # Initialize W&B logging
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    if cfg.wandb.enable:
        key = os.environ.get("WANDB_API_KEY")
        wandb_run = initialize_wandb(cfg, key=key)
        wandb_run.define_metric("epoch", summary="max")
        log_to_wandb = {k: v for e in cfg.wandb.to_log for k, v in e.items()}
        run_id = wandb_run.id
        wandb_run.name = f"l={cfg.lambda_param}_{run_id}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Device: {device}")

    # Set up output directories
    base_dir = Path(
        cfg.output_dir,
        cfg.experiment_name,
        f"fold_{cfg.data.fold}" if cfg.data.fold is not None else "",
        run_id,
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir, result_dir=   setup_directories(base_dir)
    # Load models
    model_v, model_l, tokenizer = load_models(cfg, device)

    # Load dataset
    if cfg.experiment_name == 'coco':
        trainset, validationset, testset = load_data(model_l, cfg)
    else: 
        trainset, validationset, testset = load_medical_data(model_l, tokenizer,  cfg)
    # Load loss functions and optimizer
    classification_criterion, contrastive_criterion, optim_vision = (
        load_training_components(cfg, model_v)
    )
   

    # Load checkpoint if available
    start_epoch, previous_checkpoint = load_checkpoint(
        model_v, optim_vision, checkpoint_dir
    )
    start_time = time.time()    
    # Contrastive Tuning
    model_v, training_results, validation_results = run_training(
        cfg,
        trainset,
        validationset,
        testset,
        model_v,
        model_l,
        optim_vision,
        classification_criterion,
        contrastive_criterion,
        checkpoint_dir,
        result_dir,
        log_to_wandb,
    )
    # NOW load latest model since we want to know how much performance we have lost
    best_model_fp = Path(checkpoint_dir, f"latest.pt")
    if cfg.wandb.enable:
        wandb.save(str(best_model_fp))
    best_model_sd = torch.load(best_model_fp, weights_only=True)
    model_v.load_state_dict(best_model_sd)
    # Evaluate final model
    test_results = evaluate_final_model(
        cfg, model_v, model_l, testset, log_to_wandb, result_dir
    )

    # Save results
    save_results_dict(
        training_results, validation_results, test_results, save_dir=result_dir
    )

    print(
        f"Total time taken: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}"
    )


if __name__ == "__main__":
    main()

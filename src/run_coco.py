import wandb
import time
import datetime
import tqdm
import os
import hydra
import torch
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from src.dataset import ImageTextCOCO
from src.train_copy_cp import train_epoch, tune_epoch, test

from src.utils import *


@hydra.main(
    version_base="1.2.0",
    config_path="../config/vision_language",
    config_name="biopsy_gleason",
)
def setup_directories(cfg, base_dir):
    """Creates necessary output directories."""
    checkpoint_dir = Path(base_dir, "checkpoints")
    result_dir = Path(base_dir, "results")
    embedding_dir = Path(base_dir, "embeddings") if cfg.save_embeddings else None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    if embedding_dir:
        embedding_dir.mkdir(parents=True, exist_ok=True)

    return checkpoint_dir, result_dir, embedding_dir


def load_models(cfg, device):
    """Loads and prepares vision and language models."""
    model_v = ModelFactory(cfg.vision.model_type, cfg).get_model()
    print(f"Successfully loaded Vision Model: {cfg.vision.model_type}")

    model_l = ModelFactory(cfg.language.model_type, cfg).get_model()
    for param in model_l.parameters():
        param.requires_grad = False  # Freeze language model
    print(f"Successfully loaded Language Model: {cfg.language.model_type}")

    return model_v.to(device), model_l.to(device)


def load_data(cfg):
    """Initializes datasets using COCO annotations and JSON datasets."""
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
        None,
    )


def load_training_components(cfg, model_v):
    """Loads classification & contrastive loss functions and optimizer."""
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


def load_checkpoint(model_v, optim_vision, checkpoint_dir):
    """Loads the latest checkpoint to resume training if available."""
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


def log_and_save_results(results, phase, log_dict, log_to_wandb):
    """Helper function to log and save results, remove unnecessary keys, and update global tracking dicts."""
    results.pop("cm", None)
    results.pop("wandb_cm", None)

    if cfg.wandb.enable:
        update_log_dict(
            phase,
            results,
            log_dict,
            to_log=[e for e in log_to_wandb[phase] if "cm" not in e],
            step="epoch",
        )


def clean_previous_checkpoint(previous_checkpoint, new_checkpoint):
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


# Training script
def run_training(
    cfg,
    trainset,
    validationset,
    testset,
    model_v,
    model_l,
    classification_criterion,
    contrastive_criterion,
    checkpoint_dir,
    result_dir,
    log_to_wandb,
    fold,
    early_stopping,
):
    """Handles training, validation, and logging."""

    # save results as dict
    final_training_results = {
        str(epoch): None for epoch in np.arange(cfg.training.num_epochs)
    }
    final_validation_results = {
        str(epoch): None for epoch in np.arange(cfg.training.num_epochs)
    }

    previous_checkpoint = None
    stop = False

    with tqdm.tqdm(
        range(num_epochs),
        desc="Contrastive Tuning",
        unit="epoch",
        ncols=100,
        leave=False,
    ) as t:
        for epoch in t:
            epoch_start_time = time.time()
            log_dict = {"epoch": epoch} if cfg.wandb.enable else {}

            if epoch == 0:
                # baseline validationset
                baseline_validation = tune_epoch(
                    epoch,
                    validationset,
                    model_v,
                    model_l,
                    classification_criterion,
                    contrastive_criterion,
                    config=cfg,
                )
                log_and_save_results(
                    epoch, baseline_validation, "validation", log_dict, result_dir
                )
                final_validation_results[str(epoch)] = baseline_validation
                save_json(
                    baseline_validation, result_dir, name="baseline_validation.json"
                )

                # baseline testset
                baseline_test = tune_epoch(
                    epoch,
                    testset,
                    model_v,
                    model_l,
                    classification_criterion,
                    contrastive_criterion,
                    config=cfg,
                )
                plot_cm(
                    baseline_test, log_to_wandb, result_dir, "test", wandb_enable=False
                )
                log_and_save_results(epoch, baseline_test, "test", log_dict, result_dir)
                save_json(baseline_validation, result_dir, name="baseline_testset.json")

                # logging
                if cfg.wandb.enable:
                    wandb.log(log_dict, step=epoch)

            else:
                # train epoch
                train_results, checkpoint = train_epoch(
                    epoch,
                    fold,
                    trainset,
                    model_v,
                    model_l,
                    classification_criterion,
                    contrastive_criterion,
                    checkpoint_dir=checkpoint_dir,
                )
                previous_checkpoint = clean_previous_checkpoint(
                    previous_checkpoint, checkpoint
                )
                final_training_results[str(epoch)] = train_results

                log_and_save_results(
                    epoch, train_results, "train", log_dict, result_dir
                )

                # validation epoch
                validation_results = tune_epoch(
                    epoch,
                    fold,
                    validationset,
                    model_v,
                    model_l,
                    classification_criterion,
                    contrastive_criterion,
                    phase="validation",
                    config=cfg,
                )
                final_validation_results[str(epoch)] = validation_results

                log_and_save_results(
                    epoch, validation_results, "validation", log_dict, result_dir
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
                f"End of epoch {epoch}/{num_epochs} \t Time Taken: {time.time() - epoch_start_time:.2f}s"
            )

            if stop:
                break

    return model_v, final_training_results, final_validation_results


def evaluate_final_model(
    cfg,
    model_v,
    model_l,
    testset,
    log_to_wandb,
    classification_criterion,
    contrastive_criterion,
    result_dir,
    embedding_dir,
    wandb_run,
):
    """Evaluates the final model on the test set."""
    test_results = test(
        testset,
        model_v,
        model_l,
        classification_criterion,
        contrastive_criterion,
        config=cfg,
    )
    plot_cm(
        test_results, log_to_wandb, result_dir, "test", wandb_enable=cfg.wandb.enable
    )
    save_json(test_results, result_dir, name="test_results.json")
    if cfg.save_embeddings:
        save_embeddings(testset, model_v, embedding_dir, config=cfg)

    if cfg.wandb.enable:
        wandb.log({"test": test_results}, step=cfg.training.num_epochs)

    return test_results


def main(cfg):
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

    checkpoint_dir, result_dir, embedding_dir = setup_directories(cfg, base_dir)

    # Load models
    model_v, model_l = load_models(cfg, device)

    # Load dataset
    trainset, validationset, testset = load_data(cfg)

    # Load loss functions and optimizer
    classification_criterion, contrastive_criterion, optim_vision = (
        load_training_components(cfg, model_v)
    )
    #
    early_stopping = EarlyStopping(
        cfg.early_stopping.metric,
        cfg.early_stopping.mode,
        cfg.early_stopping.patience,
        cfg.early_stopping.min_epoch,
        checkpoint_dir=checkpoint_dir,
        save_all=cfg.early_stopping.save_all,
    )

    # Load checkpoint if available
    start_epoch, previous_checkpoint = load_checkpoint(
        model_v, optim_vision, checkpoint_dir
    )

    # Training loop
    model_v, training_results, validation_results = run_training(
        cfg,
        trainset,
        validationset,
        testset,
        model_v,
        model_l,
        classification_criterion,
        contrastive_criterion,
        checkpoint_dir,
        result_dir,
        embedding_dir,
        log_to_wandb,
        cfg.data.fold,
        early_stopping,
    )
    # NOW load latest model since we want to know how much performance we have lost
    best_model_fp = Path(checkpoint_dir, f"latest.pt")
    if cfg.wandb.enable:
        wandb.save(str(best_model_fp))
    best_model_sd = torch.load(best_model_fp, weights_only=True)
    model_v.load_state_dict(best_model_sd)

    # Evaluate final model
    test_results = evaluate_final_model(
        cfg, model_v, testset, result_dir, embedding_dir, wandb_run
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

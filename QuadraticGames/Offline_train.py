"""
Offline training entrypoint for the quadratic-game regressor using DCPA approach.

This script loads pre-generated `.npz` train/validation shards containing X, Z, y,
builds the quadratic neural network, trains it from a YAML configuration, and saves
the trained weights together with loss and validation plots.

The DCPA approach:
- X: exploration features from Gaussian sampling
- Z: loss path trajectory from optimal agent
- y: optimal gradient labels

Training uses a custom DCPA loss that approximates gradients using the loss path.

Usage examples
--------------
python QuadraticGames/Offline_train.py --config QuadraticGames/Training_Data/N20/train_config.yaml --input_dir QuadraticGames/Training_Data/N20 --output_dir QuadraticGames/Training_Data/N20/results
python QuadraticGames/Offline_train.py --config path/to/train_config.yaml --input_dir path/to/dataset --output_dir path/to/results
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dnn_utils.dnn_plots import plot_loss, save_validation_scatter, save_qnn_bn_scatter
from dnn_utils.nn_utils import (
    build_loss,
    build_optimizer,
    build_scheduler,
    fit,
    get_io_dimensions,
    predict_dataset,
)
from dnn_utils.quadratic_nn import Quadratic_NN
from dnn_utils.quadratic_paths import copy_config_file, load_dataset_npz, save_model_weights
from dnn_utils.train_data_struct import XYDataset, XZYDataset, build_parser, load_configs_from_yaml


def main_train_loop(args, train_cfg, sched_cfg) -> None:
    """Run the full offline training loop and save the training artifacts."""
    X_train, Z_train, y_train, params_train, X_valid, Z_valid, y_valid, params_valid = load_dataset_npz(base_dir=args.input_dir)

    # Determine if we're using DCPA loss
    use_dcpa = train_cfg.criterion.lower() == "dcpa"
    
    if use_dcpa:
        train_dataset = XZYDataset(X_train, Z_train, y_train)
        val_dataset = XZYDataset(X_valid, Z_valid, y_valid)
    else:
        train_dataset = XYDataset(X_train, y_train)
        val_dataset = XYDataset(X_valid, y_valid)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)

    input_dim, output_dim = get_io_dimensions(train_cfg)
    print(f"input_dim: {input_dim} , output_dim: {output_dim}")

    model = Quadratic_NN(input_size=input_dim, output_size=output_dim).to(device)
    optimizer = build_optimizer(model, train_cfg)
    criterion = build_loss(train_cfg)
    scheduler = build_scheduler(optimizer, sched_cfg)

    train_list, valid_list = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=train_cfg.epochs,
        scheduler=scheduler,
        grad_clip=train_cfg.grad_clip,
        use_dcpa=use_dcpa,
    )

    save_path = save_model_weights(model, args.output_dir, "model.pt")
    cfg_copy_path = copy_config_file(args.config, args.output_dir, filename="train_config.yaml")
    print(f"Save model: {save_path}")
    print(f"Save config: {cfg_copy_path}")

    plot_loss(args.output_dir, train_cfg.epochs, train_list, valid_list)
    predictions, targets = predict_dataset(model, val_loader, device, use_dcpa=use_dcpa)
    save_validation_scatter(
        predictions=predictions,
        targets=targets,
        output_dir=args.output_dir,
        filename="val_scatter.png",
        jump=10,
    )
    
    # Save q_nn and b_n scatter plots if params are available
    if params_valid is not None:
        print("Saving q_nn and b_n scatter plots...")
        save_qnn_bn_scatter(
            predictions=predictions,
            true_params=params_valid,
            output_dir=args.output_dir,
            filename="qnn_bn_scatter.png",
            jump=10,
        )
        print("q_nn and b_n scatter plots saved!")
    else:
        print("Warning: params not available in dataset. Skipping q_nn/b_n scatter plots.")


if __name__ == "__main__":
    args = build_parser().parse_args()
    train_cfg, sched_cfg = load_configs_from_yaml(args.config)

    print("input_dir :", args.input_dir)
    print("output_dir:", args.output_dir)
    print("train_cfg :", train_cfg)
    print("sched_cfg :", sched_cfg)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_train_loop(args, train_cfg, sched_cfg)

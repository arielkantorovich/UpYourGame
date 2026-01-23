"""
Created on : ------

@author: Ariel_Kantorovich
"""

from Wireless_K.DNN_common.wireless_paths import *
from Wireless_K.DNN_common.NN_common import *
from Wireless_K.DNN_common.Train_DataStruct import *
from Wireless_K.DNN_common.wireless_NN import Wireless_NN
from Wireless_K.DNN_common.NN_plots import *
from torch.utils.data import DataLoader

def main_train_loop(args: argparse.ArgumentParser, train_cfg: TrainConfig, sched_cfg: SchedulerCfg):
    """
    Main training loop
    :param args:
    :param train_cfg:
    :param sched_cfg:
    :return:
    """
    X_train, Z_train, y_train, X_valid, Z_valid, y_valid = load_dataset_npz(base_dir=args.input_dir)

    train_dataset = XZYDataset(X_train, Z_train, y_train)
    val_dataset = XZYDataset(X_valid, Z_valid, y_valid)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print("train_loader:", train_loader)
    print("val_loader:", val_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)

    input_dim, output_dim = get_IO_NN(train_cfg.isAlphaBeta, train_cfg)
    print(f"input_dim: {input_dim} , output_dim: {output_dim}")

    model = Wireless_NN(input_size=input_dim, output_size=output_dim).to(device)
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
        is_alpha_beta=train_cfg.isAlphaBeta,
        K=train_cfg.K,
        T=train_cfg.T,
        scheduler=scheduler,  # can be None
    )

    save_path = save_model_weights(model, args.output_dir, args.weights_name)
    print(f"Save model: {save_path}")

    _ = plot_loss(args.output_dir, train_cfg.epochs, train_list, valid_list)
    predicted_prior_fn = make_predicted_prior_fn(is_alpha_beta=train_cfg.isAlphaBeta, K=train_cfg.K, T=train_cfg.T)
    _ = scatter_path = save_validation_scatter(
                                                model=model,
                                                val_loader=val_loader,
                                                device=device,
                                                predicted_prior_fn=predicted_prior_fn,
                                                output_dir=args.output_dir,
                                                filename="val_scatter.png",
                                                jump=600,
                                            )



if __name__ == "__main__":
    args = build_parser().parse_args()

    train_cfg, sched_cfg = load_configs_from_yaml(args.config)

    # Example: print what you loaded
    print("input_dir :", args.input_dir)
    print("output_dir:", args.output_dir)
    print("train_cfg :", train_cfg)
    print("sched_cfg :", sched_cfg)

    main_train_loop(args, train_cfg, sched_cfg)
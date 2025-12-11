import argparse
import os
import pandas as pd
import numpy as np
import albumentations as A

from torch.utils.data import DataLoader
# from torch.nn.parallel import DistributedDataParallel
# from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import segmentation_models_pytorch as smp
import torch

from etci_dataset import ETCIDataset
# from utils import sampler_utils
from utils import dataset_utils
from utils import metric_utils
from utils import worker_utils
import config

import warnings
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# fix all the seeds and disable non-deterministic CUDA backends for
# reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# set up logging
import logging

logging.basicConfig(
    filename="supervised_training.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


# | vv_image_path             | vh_image_path             | flood_label_path                 | water_body_label_path                 | region   |
# | ------------------------- | ------------------------- | -------------------------------- | ------------------------------------- | -------- |
# | .../vv/RedRiver_...vv.png | .../vh/RedRiver_...vh.png | .../flood_label/RedRiver_....png | .../water_body_label/RedRiver_....png | RedRiver |
def get_dataloader(rank, world_size, round):
    """Creates the data loaders."""
    # create dataframes
    train_df = dataset_utils.create_df(config.train_dir)
    valid_df = dataset_utils.create_df(config.valid_dir)

    if round > 0:
        # this path depends on where you have serialized the dataframe while
        # executing `notebook/Generate_Pseudo.ipynb`.
        print("Combining the training set with pseudo-labeled data...")
        pseudo_df = pd.read_csv("pseudo_df.csv")
        train_df = pd.concat([train_df, pseudo_df], ignore_index=True)
        print("Combined dataframe has", len(train_df), "entries.")

    # determine if an image has mask or not
    label_paths = train_df["label_path"].values.tolist()
    train_has_masks = list(map(dataset_utils.has_mask, label_paths))
    train_df["has_mask"] = train_has_masks

    # filter invalid images
    remove_indices = dataset_utils.filter_df(train_df)
    train_df = train_df.drop(train_df.index[remove_indices])

    # define augmentation transforms
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(270),
            A.ElasticTransform(
                p=0.4, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
            ),
        ]
    )

    # define datasets
    train_dataset = ETCIDataset(train_df, split="train", transform=transform)
    validation_dataset = ETCIDataset(valid_df, split="validation", transform=None)

    # create samplers
    # stratified_sampler = sampler_utils.BalanceClassSampler(
    #     train_df["has_mask"].values.astype("int")
    # )
    # train_sampler = sampler_utils.DistributedSamplerWrapper(
    #     stratified_sampler, rank=rank, num_replicas=world_size, shuffle=True
    # )
    # val_sampler = DistributedSampler(
    #     validation_dataset, rank=rank, num_replicas=world_size, shuffle=False
    # )

    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.local_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_utils.seed_worker,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=config.local_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
    )

    print("Data loaders created.")
    return train_loader, val_loader


def evaluate(model, data_loader, rank):
    model.eval()

    iou_list, dice_list, prec_list, rec_list = [], [], [], []
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            image = batch["image"].cuda(rank, non_blocking=True)
            mask = batch["mask"].cuda(rank, non_blocking=True)

            pred = model(image)
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)  # (B, H, W)

            for p, t in zip(pred, mask):
                iou_list.append(metric_utils.iou_score(p, t).cpu().item())
                dice_list.append(metric_utils.dice_score(p, t).cpu().item())
                prec_list.append(metric_utils.precision(p, t).cpu().item())
                rec_list.append(metric_utils.recall(p, t).cpu().item())

                all_true.append(t.flatten().cpu().numpy())
                all_pred.append(p.flatten().cpu().numpy())

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    return {
        "iou": np.mean(iou_list),
        "dice": np.mean(dice_list),
        "precision": np.mean(prec_list),
        "recall": np.mean(rec_list),
        "true_pixels": all_true,
        "pred_pixels": all_pred,
    }


def save_confusion_matrix(y_true, y_pred, out_path):
    """
    y_true, y_pred: flattened arrays of 0/1 pixels
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Water", "Water"],
                yticklabels=["No Water", "Water"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_loss_curve(train_losses, val_losses, out_path):
    plt.figure(figsize=(6,5))
    plt.plot(train_losses, label="Train Loss")
    if val_losses:
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    

def create_model(model_class, backbone):
    """Initializes a segmentation model.

    Args:
        model_class: One of smp.Unet, smp.UnetPlusPlus
        backbone: Encoder backbone. Should always be "mobilenet_v2"
                    for this purpose.

    """
    if model_class == "unet":
        model_class = smp.Unet
    elif model_class == "unetplusplus":
        model_class = smp.UnetPlusPlus
    else:
        raise ValueError("model_class should be one of 'unet' or 'unetplusplus'")
    
    model = model_class(
        encoder_name=backbone, encoder_weights="imagenet", in_channels=3, classes=2
    )
    return model


def train(rank, model_class, num_epochs, world_size, round, eval=False):
    """Trains the segmentation model using distributed training."""
    # initialize the workers and fix the seeds
    worker_utils.init_process(rank, world_size)
    torch.manual_seed(0)

    # model loading and off-loading to the current device
    model = create_model(model_class, config.backbone)
    torch.cuda.set_device(rank)
    model.cuda(rank)
    # model = DistributedDataParallel(model, device_ids=[rank])
    model = model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # set up loss function and gradient scaler for mixed-precision
    criterion = smp.losses.TverskyLoss(
        mode="multiclass",
        alpha=0.7,
        beta=0.3
    ) + 0.5 * smp.losses.FocalLoss("binary")
    scaler = torch.amp.GradScaler(enabled=True)

    # initialize data loaders
    train_loader, val_loader = get_dataloader(rank, world_size, round)

    ## begin training ##
    print("Starting training...")
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        losses = metric_utils.AvgMeter()

        if rank == 0:
            logging.info(
                "Rank: {}/{} Epoch: [{}/{}]".format(
                    rank, world_size, epoch + 1, num_epochs
                )
            )

        # train set
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training"):
            with torch.amp.autocast(device_type="cuda", enabled=True):
                image = batch["image"].cuda(rank, non_blocking=True)
                mask = batch["mask"].cuda(rank, non_blocking=True)
                pred = model(image)

                loss = criterion(pred, mask)
                losses.update(loss.cpu().item(), image.size(0))

            # update the model
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # average out the losses and log the summary
        loss = losses.avg
        train_losses.append(loss)
        # global_loss = metric_utils.global_meters_all_avg(rank, world_size, loss)

        if rank == 0:
            logging.info(f"Epoch: {epoch+1} Train Loss: {loss:.3f}")

        ## evaluation ##
        if epoch % 5 == 0:
            if rank == 0:
                logging.info("Running evaluation on the validation set.")
            model.eval()
            losses = metric_utils.AvgMeter()

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Validation"):
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        image = batch["image"].cuda(rank, non_blocking=True)
                        mask = batch["mask"].cuda(rank, non_blocking=True)
                        pred = model(image)

                        loss = criterion(pred, mask)
                        losses.update(loss.cpu().item(), image.size(0))

            loss = losses.avg
            val_losses.append(loss)
            # global_loss = metric_utils.global_meters_all_avg(rank, world_size, loss)

            if rank == 0:
                logging.info(f"Epoch: {epoch+1} Val Loss: {loss:.3f}")

    # serialization of model weights
    out_dir = f"src/model/round_{round}/{model_class}_{config.backbone}"
    os.makedirs(out_dir, exist_ok=True)
    if rank == 0:
        torch.save(
            model.state_dict(), f"{out_dir}/saved_weights.pth"
        )
        print("Model weights saved.")
    print("Training complete.")

    # plot loss curves
    plot_loss_curve(train_losses, val_losses, f"{out_dir}/loss_curve.png")

    if eval:
        print("Starting evaluation...")
        
        # final evaluation on train and val sets
        train_metrics = evaluate(model, train_loader, rank)
        print("TRAIN METRICS:", train_metrics)

        val_metrics = evaluate(model, val_loader, rank)
        print("VAL METRICS:", val_metrics)

        # plot confusion matrices
        save_confusion_matrix(
            train_metrics["true_pixels"],
            train_metrics["pred_pixels"],
            f"{out_dir}/train_confusion_matrix.png")

        save_confusion_matrix(
            val_metrics["true_pixels"],
            val_metrics["pred_pixels"],
            f"{out_dir}/val_confusion_matrix.png")
        print("Evaluation complete.")


WORLD_SIZE = torch.cuda.device_count()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_class", type=str, default="unet", help="unet or unetplusplus")
    ap.add_argument("-e", "--epochs", type=int, default=15, help="number of epochs")
    ap.add_argument("-r", "--round", type=int, default=0, help="round of pseudo-labeling")
    ap.add_argument("--eval", action="store_true", help="Run evaluation instead of training")
    args = vars(ap.parse_args())

    # mp.spawn(train, args=(config.num_epochs, WORLD_SIZE), nprocs=WORLD_SIZE, join=True)

    train(rank=0,
          model_class=args["model_class"],
          num_epochs=args["epochs"],
          world_size=WORLD_SIZE,
          round=args["round"],
          eval=args["eval"])
    
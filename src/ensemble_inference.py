import os
from glob import glob
import warnings
import subprocess
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchvision
import requests

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import ttach as tta
import argparse

from utils import dataset_utils
from utils import metric_utils
import config

warnings.filterwarnings("ignore")


# ---------- Utility Functions ----------
def get_test_id(path):
    return path.split("_")[0] + "_" + path.split("_")[1]


def make_im_name(id, suffix):
    return id.split(".")[0] + f"_{suffix}.png"


def s1_to_rgb(vv_image, vh_image):
    ratio_image = np.clip(np.nan_to_num(vh_image / vv_image, 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1 - ratio_image), axis=2)
    return rgb_image


def matplotlib_imshow(img):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img.permute(1, 2, 0).numpy())
    plt.show()


# ---------- Dataset ----------
class ETCIDataset(Dataset):
    def __init__(self, dataframe, split, transform=None):
        self.split = split
        self.dataset = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        df_row = self.dataset.iloc[index]

        vv_image = cv2.imread(df_row["vv_image_path"], 0) / 255.0
        vh_image = cv2.imread(df_row["vh_image_path"], 0) / 255.0

        rgb_image = s1_to_rgb(vv_image, vh_image)
        mask = cv2.imread(df_row["label_path"], 0) / 255.0

        if self.split == "train":
            if self.transform:
                augmented = self.transform(image=rgb_image, mask=mask)
                rgb_image = augmented["image"]
                mask = augmented["mask"]
        return {
            "image": rgb_image.transpose((2, 0, 1)).astype("float32"),
            "mask": mask.astype("int64"),
        }


# ---------- Model Factory ----------
def get_model_by_name(name):
    """
    Maps command-line model names to instantiated SMP models.
    """
    name = name.lower()

    if name == "unet_mobilenet_v2":
        return smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=2
        )

    elif name in ("unetplusplus_mobilenet_v2", "unetpp_mobilenet_v2", "upp_mobilenet_v2"):
        return smp.UnetPlusPlus(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=2
        )

    else:
        raise ValueError(f"Unknown model name: {name}")


# ---------- Prediction Function ----------
def get_predictions_single(model_def, weight_path, test_loader, device):
    model_def.load_state_dict(torch.load(weight_path, map_location=device))
    model = tta.SegmentationTTAWrapper(
        model_def, tta.aliases.d4_transform(), merge_mode="mean"
    )
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    preds_all = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Predicting with {weight_path}"):
            imgs = batch["image"].to(device)
            pred = model(imgs)
            preds_all.append(pred)

    preds_all = torch.cat(preds_all, dim=0) # (N, 2, H, W)
    return preds_all


def evaluate_ensemble(preds, dataset, device, rank=0):
    """
    preds: tensor of shape (N, 2, H, W)
    dataset: ETCIDataset(split='test')
    """
    loader = DataLoader(dataset, batch_size=96, shuffle=False)

    iou_list, dice_list, prec_list, rec_list = [], [], [], []
    all_true, all_pred = [], []

    idx_start = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Ensemble"):
            # slice ensemble output
            B = batch["mask"].shape[0]
            batch_preds = preds[idx_start:idx_start+B]
            idx_start += B

            # process predictions
            batch_preds = torch.softmax(batch_preds, dim=1)
            batch_preds = torch.argmax(batch_preds, dim=1) # (B, H, W)

            masks = batch["mask"].to(device)

            batch_preds = batch_preds.long()
            masks = masks.long()

            for p, t in zip(batch_preds, masks):
                iou_list.append(metric_utils.iou_score(p, t).item())
                dice_list.append(metric_utils.dice_score(p, t).item())
                prec_list.append(metric_utils.precision(p, t).item())
                rec_list.append(metric_utils.recall(p, t).item())

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
        "pred_pixels": all_pred
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


# ---------- Main ----------
def main(args):
    # Parse CLI lists
    model_names = [m.strip() for m in args.model_defs.split(",")]
    model_paths = [p.strip() for p in args.model_paths.split(",")]

    assert len(model_names) == len(model_paths), \
        "model-defs and model-paths must be the same length."

    # Instantiate models
    model_defs = [get_model_by_name(name) for name in model_names]

    # Dataset root
    dset_root = "final-UNOSAT-Dataset/data/"
    test_dir = os.path.join(dset_root, "test_internal")

    print("Number of test temporal-regions:", len(glob(test_dir + "/*/")))

    # Load CSV list
    # url = "https://git.io/JsRTE"
    # r = requests.get(url)
    # with open("test_sentinel.csv", "wb") as f:
    #     f.write(r.content)
    # print("Downloaded test_sentinel.csv to", os.path.abspath("test_sentinel.csv"))

    # test_file_sequence = (
    #     pd.read_csv("test_sentinel.csv", header=None)
    #     .values.squeeze().tolist()
    # )

    # all_test_vv = [
    #     os.path.join(test_dir, get_test_id(id), "tiles", "vv", make_im_name(id, "vv"))
    #     for id in test_file_sequence
    # ]
    # all_test_vh = [
    #     os.path.join(test_dir, get_test_id(id), "tiles", "vh", make_im_name(id, "vh"))
    #     for id in test_file_sequence
    # ]

    # test_df = pd.DataFrame({"vv_image_path": all_test_vv, "vh_image_path": all_test_vh})
    # print(test_df.shape)
    test_df = dataset_utils.create_df(config.test_dir)

    # Dataset + loader
    test_dataset = ETCIDataset(test_df, split="test")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 96 * max(1, torch.cuda.device_count())

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Visual sample (optional)
    images = next(iter(test_loader))["image"]
    img_grid = torchvision.utils.make_grid(images[50:59], nrow=3)
    matplotlib_imshow(img_grid)

    # Run predictions
    all_preds = []
    for model_def, path in zip(model_defs, model_paths):
        preds = get_predictions_single(model_def, path, test_loader, device)
        all_preds.append(preds)

    # # Ensemble
    # all_preds = np.array(all_preds)
    # all_preds = np.mean(all_preds, axis=0)
    # class_preds = all_preds.argmax(axis=1).astype("uint8")

    # # Save submission
    # np.save(args.submission_path, class_preds, fix_imports=True, allow_pickle=False)
    # subprocess.run(["zip", args.zip_name, args.submission_path])

    # print(f"Saved {args.zip_name}")

    # metrics_U = evaluate_ensemble(all_preds[0], test_dataset, device)
    # print("ENSEMBLE METRICS:")
    # for k, v in metrics_U.items():
    #     if isinstance(v, float):
    #         print(f"{k}: {v:.4f}")

    # metrics_PP = evaluate_ensemble(all_preds[1], test_dataset, device)
    # print("ENSEMBLE METRICS:")
    # for k, v in metrics_PP.items():
    #     if isinstance(v, float):
    #         print(f"{k}: {v:.4f}")

    ensemble_logits = torch.mean(torch.stack(all_preds), dim=0) # (2, N, 2, H, W)
    metrics = evaluate_ensemble(ensemble_logits, test_dataset, device)

    print("ENSEMBLE METRICS:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")

    out_dir = f"src/model/round_{args.round}"
    os.makedirs(out_dir, exist_ok=True)
    save_confusion_matrix(
        metrics["true_pixels"],
        metrics["pred_pixels"],
        f"{out_dir}/ensemble_confusion_matrix.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETCI Flood Segmentation Inference")

    parser.add_argument(
        "--model-defs",
        type=str,
        required=True,
        help="Comma-separated model names (e.g., unet_mobilenet_v2,upp_mobilenet_v2)",
    )
    parser.add_argument(
        "--model-paths",
        type=str,
        required=True,
        help="Comma-separated model weight paths",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=0,
        help="round of pseudo-labeling"
    )

    args = parser.parse_args()
    main(args)
    
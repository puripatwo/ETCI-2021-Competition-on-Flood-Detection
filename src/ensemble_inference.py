import os
from glob import glob
import warnings
import subprocess
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import requests
import argparse

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import ttach as tta

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


# ---------- Dataset ----------
class ETCIDataset(Dataset):
    def __init__(self, dataframe, split, transform=None):
        self.split = split
        self.dataset = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        example = {}

        df_row = self.dataset.iloc[index]
        vv_image = cv2.imread(df_row["vv_image_path"], 0) / 255.0
        vh_image = cv2.imread(df_row["vh_image_path"], 0) / 255.0

        rgb_image = s1_to_rgb(vv_image, vh_image)

        if self.split == "test":
            example["image"] = rgb_image.transpose((2, 0, 1)).astype("float32")
        else:
            flood_mask = cv2.imread(df_row["flood_label_path"], 0) / 255.0

            if self.transform:
                augmented = self.transform(image=rgb_image, mask=flood_mask)
                rgb_image = augmented["image"]
                flood_mask = augmented["mask"]

            example["image"] = rgb_image.transpose((2, 0, 1)).astype("float32")
            example["mask"] = flood_mask.astype("int64")

        return example


# ---------- Prediction Function ----------
def get_predictions_single(model_def, weight_path, test_loader, device):
    model_def.load_state_dict(torch.load(weight_path))
    model = tta.SegmentationTTAWrapper(
        model_def,
        tta.aliases.d4_transform(),
        merge_mode="mean"
    )
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    final_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            imgs = batch["image"].to(device)
            pred = model(imgs)
            final_preds.append(pred.detach().cpu().numpy())

    return np.concatenate(final_preds, axis=0)


# ---------- Model Making Function ----------
def make_model(model_type: str):
    """Map string name to actual SMP model."""
    if model_type == "unet_mobilenet":
        return smp.Unet(
            encoder_name="mobilenet_v2", encoder_weights=None,
            in_channels=3, classes=2
        )
    elif model_type == "unetplusplus_mobilenet":
        return smp.UnetPlusPlus(
            encoder_name="mobilenet_v2", encoder_weights=None,
            in_channels=3, classes=2
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------- Main ----------
def main(args):
    # Load test images
    dset_root = "ETCI-2021-Flood-Detection/data/"
    test_dir = os.path.join(dset_root, "test_internal")

    url = "https://git.io/JsRTE"
    r = requests.get(url)
    with open("test_sentinel.csv", "wb") as f:
        f.write(r.content)

    test_file_sequence = (
        pd.read_csv("test_sentinel.csv", header=None)
        .values.squeeze().tolist()
    )

    all_test_vv = [
        os.path.join(test_dir, get_test_id(id), "tiles", "vv", make_im_name(id, "vv"))
        for id in test_file_sequence
    ]
    all_test_vh = [
        os.path.join(test_dir, get_test_id(id), "tiles", "vh", make_im_name(id, "vh"))
        for id in test_file_sequence
    ]

    test_df = pd.DataFrame({"vv_image_path": all_test_vv, "vh_image_path": all_test_vh})

    # DataLoader
    batch_size = 96 * torch.cuda.device_count()
    dataset = ETCIDataset(test_df, split="test", transform=None)

    test_loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=os.cpu_count(),
        pin_memory=True
    )

    # Models
    model_paths = args.model_paths
    model_types = args.model_types

    if len(model_paths) != len(model_types):
        raise ValueError("Number of --model-paths must match --model-types.")

    print(f"Using {len(model_paths)} models:")
    for m, p in zip(model_types, model_paths):
        print(f" - {m} : {p}")

    # Load and run models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_preds = []

    for mtype, path in zip(model_types, model_paths):
        model = make_model(mtype)
        preds = get_predictions_single(model, path, test_loader, device)
        all_preds.append(preds)

    # Ensemble and save
    all_preds = np.array(all_preds)
    all_preds = np.mean(all_preds, axis=0)
    class_preds = all_preds.argmax(axis=1).astype("uint8")

    save_path = args.output
    np.save(save_path, class_preds, fix_imports=True, allow_pickle=False)

    subprocess.run(["zip", "submission.zip", save_path])
    print(f"Saved submission.zip containing {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETCI Flood Prediction")

    parser.add_argument(
        "--model-paths", nargs="+", type=str, required=False,
        default=[
            "src/model/round_0/unet_mobilenet_v2_0.pth",
            "src/model/round_0/upp_mobilenet_v2_0.pth",
        ],
        help="List of .pth weight files"
    )

    parser.add_argument(
        "--model-types", nargs="+", type=str, required=False,
        default=[
            "unet_mobilenet",
            "unetplusplus_mobilenet",
        ],
        help="List of model architectures matching model-paths"
    )

    parser.add_argument(
        "--output", type=str, default="submission.npy",
        help="Numpy file to save predictions"
    )

    args = parser.parse_args()
    main(args)

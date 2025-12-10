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

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import ttach as tta
import argparse

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
### TODO ###
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

        if self.split == "test":
            return {"image": rgb_image.transpose((2, 0, 1)).astype("float32")}

        else:
            flood_mask = cv2.imread(df_row["flood_label_path"], 0) / 255.0
            if self.transform:
                augmented = self.transform(image=rgb_image, mask=flood_mask)
                rgb_image = augmented["image"]
                flood_mask = augmented["mask"]
            return {
                "image": rgb_image.transpose((2, 0, 1)).astype("float32"),
                "mask": flood_mask.astype("int64"),
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
    model_def.load_state_dict(torch.load(weight_path))
    model = tta.SegmentationTTAWrapper(
        model_def, tta.aliases.d4_transform(), merge_mode="mean"
    )
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()
    final_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Predicting with {weight_path}", leave=True, dynamic_ncols=True):
            imgs = batch["image"].to(device)
            pred = model(imgs)
            final_preds.append(pred.detach().cpu().numpy())

    return np.concatenate(final_preds, axis=0)


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
    dset_root = "final-ETCI-2021-Flood-Detection/data/"
    test_dir = os.path.join(dset_root, "test_internal")

    print("Number of test temporal-regions:", len(glob(test_dir + "/*/")))

    # Load CSV list
    ### TODO ###
    url = "https://git.io/JsRTE"
    r = requests.get(url)
    with open("test_sentinel.csv", "wb") as f:
        f.write(r.content)
    print("Downloaded test_sentinel.csv to", os.path.abspath("test_sentinel.csv"))

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
    print(test_df.shape)

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

    # Ensemble
    all_preds = np.array(all_preds)
    all_preds = np.mean(all_preds, axis=0)
    class_preds = all_preds.argmax(axis=1).astype("uint8")

    # Save submission
    np.save(args.submission_path, class_preds, fix_imports=True, allow_pickle=False)
    subprocess.run(["zip", args.zip_name, args.submission_path])

    print(f"Saved {args.zip_name}")


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
        "--submission-path",
        type=str,
        default="submission.npy",
        help="Output .npy file for predictions",
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default="submission.zip",
        help="Name of output zip archive",
    )

    args = parser.parse_args()
    main(args)
    
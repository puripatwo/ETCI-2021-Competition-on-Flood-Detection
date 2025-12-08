import os
import sys
from glob import glob
import warnings
import cv2
import subprocess
import numpy as np
import pandas as pd
import requests
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import ttach as tta
import segmentation_models_pytorch as smp

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

        # load vv and vh images
        vv_image = cv2.imread(df_row['vv_image_path'], 0) / 255.0
        vh_image = cv2.imread(df_row['vh_image_path'], 0) / 255.0

        # convert vv and vh images to rgb
        rgb_image = s1_to_rgb(vv_image, vh_image)

        if self.split == 'test':
            example['image'] = rgb_image.transpose((2, 0, 1)).astype('float32')
            example['vv_image_path'] = df_row['vv_image_path']
            example['vh_image_path'] = df_row['vh_image_path']

        else:
            flood_mask = cv2.imread(df_row['flood_label_path'], 0) / 255.0

            if self.transform:
                augmented = self.transform(image=rgb_image, mask=flood_mask)
                rgb_image = augmented['image']
                flood_mask = augmented['mask']

            example['image'] = rgb_image.transpose((2, 0, 1)).astype("float32")
            example['mask'] = flood_mask.astype("int64")

        return example


# ---------- Prediction Function ----------
def get_predictions_single(model_defs, weights, dir_path, test_loader, device,
                           conf_thres=0.95, pixel_thres=0.9):
    models = []

    for model_def, weight in zip(model_defs, weights):
        model_def.load_state_dict(torch.load(weight))
        model = tta.SegmentationTTAWrapper(model_def, tta.aliases.d4_transform(), merge_mode="mean")
        model.to(device)
        model.eval()

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        models.append(model)

    os.makedirs(dir_path, exist_ok=True)

    vv_s = []
    vh_s = []
    masks = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            image = batch["image"].to(device)

            preds = []
            for model in models:
                pred = model(image)
                preds.append(pred.detach().cpu().numpy())

            preds = np.array(preds)
            preds = np.mean(preds, axis=0)

            filter_preds, _ = nn.Softmax(dim=1)(torch.tensor(preds)).max(1)
            filter_preds = filter_preds.numpy()

            filtered = (
                np.sum(filter_preds > conf_thres, axis=(1, 2))
                > pixel_thres * 256 * 256
            )

            for idx, filter_ in enumerate(filtered):
                if filter_:
                    vv_s.append(batch['vv_image_path'][idx])
                    vh_s.append(batch['vh_image_path'][idx])

                    entry = nn.Softmax(dim=0)(torch.tensor(preds[idx])).argmax(0).numpy() * 255.
                    pseudo_path = "_".join(batch['vv_image_path'][idx].split("/")[-1].split("_")[:-1]) + ".png"
                    pseudo_path = os.path.join(dir_path, pseudo_path)

                    masks.append(pseudo_path)
                    cv2.imwrite(pseudo_path, entry.astype("float32"))

    return vv_s, vh_s, masks


# ---------- Main ----------
def main():
    # Dataset root
    dset_root = "ETCI-2021-Flood-Detection/data/"
    test_dir = os.path.join(dset_root, "test_internal")

    n_test_regions = len(glob(test_dir + "/*/"))
    print("Number of test temporal-regions:", n_test_regions)

    # Load CSV list
    url = "https://git.io/JsRTE"
    r = requests.get(url)
    with open("test_sentinel.csv", "wb") as f:
        f.write(r.content)

    test_file_sequence = (
        pd.read_csv("test_sentinel.csv", header=None).values.squeeze().tolist()
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

    # Dataset + Loader
    test_dataset = ETCIDataset(test_df, split="test", transform=None)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 96 * torch.cuda.device_count()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    # Models
    unet_mobilenet = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=2
    )

    upp_mobilenet = smp.UnetPlusPlus(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=2
    )

    model_defs = [unet_mobilenet, upp_mobilenet]
    model_paths = [
        "src/model/round_0/unet_mobilenet_v2_0.pth",
        "src/model/round_0/upp_mobilenet_v2_0.pth",
    ]

    # Run predictions
    vv_s, vh_s, masks = get_predictions_single(
        model_defs=model_defs,
        weights=model_paths,
        dir_path="pseudo_labels",
        test_loader=test_loader,
        device=device
    )

    assert len(vv_s) == len(vh_s) == len(masks)

    pseudo_df = pd.DataFrame(
        {
            "vv_image_path": vv_s,
            "vh_image_path": vh_s,
            "flood_label_path": masks,
        }
    )

    print(pseudo_df.shape)
    pseudo_df.to_csv("pseudo_df.csv", index=False)


if __name__ == "__main__":
    main()

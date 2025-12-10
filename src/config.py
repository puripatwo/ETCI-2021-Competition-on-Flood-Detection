import os
import segmentation_models_pytorch as smp

# dataset related
dataset_root = "final-UNOSAT-Dataset/data/"
train_dir = os.path.join(dataset_root, "train")
valid_dir = os.path.join(dataset_root, "test")
test_dir = os.path.join(dataset_root, "test_internal")
local_batch_size = 96

# model related
backbone = "mobilenet_v2"

# training related
learning_rate = 1e-3

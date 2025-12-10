"""
Referenced from:
https://discuss.pytorch.org/t/right-ways-to-serialize-and-load-ddp-model-checkpoints/122719/3
"""
import torch
import torch.distributed as dist
import numpy as np


def global_meters_all_avg(rank, world_size, *meters):
    tensors = [
        torch.tensor(meter, device=rank, dtype=torch.float32) for meter in meters
    ]
    for tensor in tensors:
        # each item of `tensors` is all-reduced starting from index 0 (in-place)
        dist.all_reduce(tensor)

    return [(tensor / world_size).item() for tensor in tensors]


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def iou_score(pred, true, eps=1e-6):
    pred = pred.bool()
    true = true.bool()
    intersection = (pred & true).sum().float()
    union = (pred | true).sum().float()
    return (intersection + eps) / (union + eps)

def dice_score(pred, true, eps=1e-6):
    pred = pred.bool()
    true = true.bool()
    intersection = (pred & true).sum().float()
    return (2 * intersection + eps) / (pred.sum() + true.sum() + eps)

def precision(pred, true, eps=1e-6):
    pred = pred.bool()
    true = true.bool()
    tp = (pred & true).sum().float()
    fp = (pred & ~true).sum().float()
    return (tp + eps) / (tp + fp + eps)

def recall(pred, true, eps=1e-6):
    pred = pred.bool()
    true = true.bool()
    tp = (pred & true).sum().float()
    fn = (~pred & true).sum().float()
    return (tp + eps) / (tp + fn + eps)

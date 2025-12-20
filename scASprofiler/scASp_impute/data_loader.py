from __future__ import annotations

from dataclasses import dataclass


from .utils import load_and_preprocess_data

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Masks:
    obs_mask: np.ndarray
    val_mask: np.ndarray


def load_labels(file_c: str) -> np.ndarray:
    """Load labels (one column). Returns integer codes 0..ncls-1."""
    s = pd.read_csv(file_c, header=None, index_col=False).iloc[:, 0]
    return pd.Categorical(s).codes


def build_validation_mask(df_raw: np.ndarray, drop_prob: float) -> Masks:
    obs_mask = ~np.isnan(df_raw)
    val_mask = obs_mask.copy()

    obs_idx = np.argwhere(obs_mask)
    drop_n = int(len(obs_idx) * drop_prob)
    if drop_n > 0:
        drop_i = obs_idx[np.random.choice(len(obs_idx), drop_n, replace=False)]
        for i, j in drop_i:
            val_mask[i, j] = False

    return Masks(obs_mask=obs_mask, val_mask=val_mask)


class ImageMatrixDataset(Dataset):
    def __init__(self, file_d, file_c, img_size, val_mask=None, transform=None):
        super().__init__()

        raw, _ = load_and_preprocess_data(file_d)
        data = raw.values

        if val_mask is None:
            mask = (~np.isnan(data)).astype(np.float32)
        else:
            data = data.copy()
            data[val_mask == False] = np.nan
            mask = val_mask.astype(np.float32)

        data = np.nan_to_num(data, nan=0.0).astype(np.float32)

        self.data_df = data
        self.mask_df = mask
        self.labels = load_labels(file_c)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, idx: int):

        x = self.data_df[:, idx].reshape(self.img_size, self.img_size, 1)
        m = self.mask_df[:, idx].reshape(self.img_size, self.img_size, 1)

        y = np.array(self.labels[idx]).astype("int32")

        sample = {"data": x, "mask": m, "label": y}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class ToTensor:
    def __call__(self, sample):
        import torch
        x, m, y = sample["data"], sample["mask"], sample["label"]

        x = torch.from_numpy(x.transpose(2, 0, 1))
        m = torch.from_numpy(m.transpose(2, 0, 1))
        y = torch.from_numpy(y)  # int32
        return {"data": x, "mask": m, "label": y}


def flatten_images_to_matrix(imgs: np.ndarray) -> np.ndarray:
    """(n_samples, 1, H, W) -> (n_features, n_samples)"""
    if imgs.ndim != 4:
        raise ValueError(f"Expected 4D array, got {imgs.shape}")
    n = imgs.shape[0]
    flat = imgs.reshape(n, -1).T
    return flat
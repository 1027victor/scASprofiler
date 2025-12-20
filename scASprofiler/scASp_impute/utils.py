from __future__ import annotations

import os
import random
import importlib
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch = importlib.import_module("torch")
    cudnn = torch.backends.cudnn

    # --- Python / NumPy / PyTorch RNG ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        cudnn.deterministic = True
        cudnn.benchmark = False

    torch.backends.cuda.matmul.allow_tf32 = False
    cudnn.allow_tf32 = False


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_job_name(file_d: str, file_c: str, job_name: str = "") -> str:
    if job_name:
        return job_name
    return f"{Path(file_d).name}-{Path(file_c).name}"


def build_model_basename(job_name: str, latent_dim: int, n_epochs: int, ncls: int) -> str:
    return f"{job_name}-{latent_dim}-{n_epochs}-{ncls}"


def one_hot(labels, depth: int):
    """Torch one-hot with the same behavior as your original helper."""
    import torch

    eye = torch.eye(depth, device=labels.device)
    return eye.index_select(0, labels)


def weights_init_normal(m) -> None:
    """Weight init matching your original script."""
    import torch

    classname = m.__class__.__name__
    if "Conv" in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm2d" in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def infer_device() -> Tuple["torch.device", bool]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = bool(torch.cuda.is_available())
    return device, cuda


def up_sample_zero(mtx, row_num):
    m_rows, m_cols = mtx.shape
    if m_rows >= row_num:
        return mtx
    else:

        nan_matrix = np.full((row_num - m_rows, m_cols), np.nan)
        nan_df = pd.DataFrame(nan_matrix, columns=mtx.columns)

        return pd.concat([mtx, nan_df], ignore_index=True)


def load_and_preprocess_data(file_d):

    merge_df = pd.read_csv(file_d, sep="\t", index_col=0)
    d = merge_df.iloc[:, :-3]

    sj_coord = d.shape[0]
    fig_h = int(np.ceil(np.sqrt(sj_coord)))

    sj_padding = up_sample_zero(d, fig_h**2)

    norm_sj = sj_padding / sj_padding.max()
    return norm_sj, fig_h


def denorm_sj(file_d, imputed_norm_sj):

    merge_df = pd.read_csv(file_d, sep="\t", index_col=0)
    d = merge_df.fillna(0).iloc[:, :-3]
    max_columns = d.max()
    vae_gan_output = imputed_norm_sj.iloc[:d.shape[0], :]
    vae_gan_output.index = d.index
    vae_gan_output.columns = d.columns
    denorm_sj = vae_gan_output * max_columns
    denorm_sj = pd.concat(
        [merge_df[["coord.intron", "intron_group", "annotated"]], denorm_sj], axis=1)
    return denorm_sj

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import IOConfig, ImputeConfig, ModelConfig
from .data_loader import load_labels
from .utils import load_and_preprocess_data
from .knn import my_knn_type
from .models import Decoder
from .utils import build_job_name, build_model_basename, infer_device, one_hot, denorm_sj


def _default_decoder_path(io: IOConfig, model_cfg: ModelConfig, n_epochs: int) -> Path:
    job_name = build_job_name(io.file_d, io.file_c, io.job_name)
    basename = build_model_basename(
        job_name, model_cfg.latent_dim, n_epochs, model_cfg.ncls)
    return Path(io.outdir) / "models" / f"{basename}-dec_best.pt"


def impute(
    io: IOConfig,
    model_cfg: ModelConfig,
    impute_cfg: ImputeConfig,
    decoder_ckpt: str | None = None,
    n_epochs_for_name: int = 200,
) -> Path:
    """Impute a matrix using the trained decoder + within-class KNN refinement."""
    device, _ = infer_device()

    decoder_path = Path(decoder_ckpt) if decoder_ckpt else _default_decoder_path(
        io, model_cfg, n_epochs_for_name)
    if not decoder_path.is_file():
        raise FileNotFoundError(
            f"Decoder checkpoint not found: {decoder_path}. "
            "Run `scasp train ...` first or pass `--decoder-ckpt` explicitly."
        )

    decoder = Decoder(
        img_size=model_cfg.img_size,
        channels=model_cfg.channels,
        latent_dim=model_cfg.latent_dim,
        ncls=model_cfg.ncls,
    ).to(device)

    state = torch.load(decoder_path, map_location=device)
    decoder.load_state_dict(state)

    raw_arr, _ = load_and_preprocess_data(io.file_d)
    raw_arr = raw_arr.values
    data_mat = np.nan_to_num(raw_arr, nan=0.0)
    mask_mat = (~np.isnan(raw_arr)).astype(np.int8)

    labels = load_labels(io.file_c)

    n_feat, n_cells = data_mat.shape
    imputed = np.zeros((n_feat, n_cells), dtype=float)

    # Generate class-conditional simulations
    sim_out: list[np.ndarray] = []
    with torch.no_grad():
        for c in range(model_cfg.ncls):
            z = torch.randn(impute_cfg.sim_size,
                            model_cfg.latent_dim, device=device)
            lbls = one_hot(torch.full((impute_cfg.sim_size,), c,
                           dtype=torch.long, device=device), model_cfg.ncls)
            fake = decoder(z, lbls)  # (sim_size,1,H,W)
            fake = fake.detach().cpu().numpy().reshape(
                impute_cfg.sim_size, -1)  # (sim_size,n_feat)
            sim_out.append(fake)

    # KNN per sample
    for j in range(n_cells):
        data_k = data_mat[:, j]
        mask_k = mask_mat[:, j]
        sim_k = sim_out[int(labels[j])]
        imputed[:, j] = my_knn_type(
            data_k, mask_k, sim_k, knn_k=impute_cfg.knn_k)

    imputed_data = denorm_sj(io.file_d, pd.DataFrame(imputed))

    job_name = build_job_name(io.file_d, io.file_c, io.job_name)

    out_dir = Path(io.outdir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"scASP-{job_name}.csv"

    imputed_data.to_csv(out_path)
    return out_path

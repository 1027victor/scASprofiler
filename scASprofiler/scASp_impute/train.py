from __future__ import annotations
from .data_loader import ToTensor

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import IOConfig, ModelConfig, TrainConfig
from .data_loader import (
    ImageMatrixDataset,
    build_validation_mask,
    load_labels,
    load_and_preprocess_data,
    flatten_images_to_matrix,
)
from .models import Discriminator, VAEGAN
from .utils import (
    build_job_name,
    build_model_basename,
    ensure_dir,
    infer_device,
    one_hot,
)


@dataclass(frozen=True)
class Checkpoints:
    enc_best: Path
    dec_best: Path
    d_best: Path


def _checkpoint_paths(outdir: str, basename: str) -> Tuple[Path, Checkpoints]:
    models_dir = ensure_dir(Path(outdir) / "models")
    ckpt = Checkpoints(
        enc_best=models_dir / f"{basename}-enc_best.pt",
        dec_best=models_dir / f"{basename}-dec_best.pt",
        d_best=models_dir / f"{basename}-d_best.pt",
    )
    return models_dir, ckpt


def train(
    io: IOConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    # seed: int = 42,
    overwrite: bool = False,
) -> Checkpoints:
    """Train encoder/decoder/discriminator and save best checkpoints."""
    device, _ = infer_device()

    job_name = build_job_name(io.file_d, io.file_c, io.job_name)
    basename = build_model_basename(
        job_name, model_cfg.latent_dim, train_cfg.n_epochs, model_cfg.ncls)

    _, ckpt = _checkpoint_paths(io.outdir, basename)

    if (ckpt.dec_best).is_file() and not overwrite:
        return ckpt

    if train_cfg.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    df_raw, _ = load_and_preprocess_data(io.file_d)
    df_raw = df_raw.values
    masks = build_validation_mask(df_raw, drop_prob=train_cfg.drop_prob)

    _, n_cells = df_raw.shape
    z_fixed = torch.randn(n_cells, model_cfg.latent_dim, device=device)
    labels = load_labels(io.file_c)
    lbls_oh = one_hot(torch.tensor(labels, dtype=torch.long,
                      device=device), model_cfg.ncls)

    # --- Models ---
    generator = VAEGAN(
        img_size=model_cfg.img_size,
        channels=model_cfg.channels,
        latent_dim=model_cfg.latent_dim,
        ncls=model_cfg.ncls,
    ).to(device)

    discriminator = Discriminator(
        img_size=model_cfg.img_size,
        channels=model_cfg.channels,
        ncls=model_cfg.ncls,
    ).to(device)

    # --- Optimizers ---
    optimizer_E = torch.optim.Adam(generator.encoder.parameters(
    ), lr=train_cfg.lr, betas=(train_cfg.b1, train_cfg.b2))
    optimizer_D = torch.optim.Adam(generator.decoder.parameters(
    ), lr=train_cfg.lr, betas=(train_cfg.b1, train_cfg.b2))
    optimizer_Dis = torch.optim.Adam(discriminator.parameters(
    ), lr=train_cfg.lr, betas=(train_cfg.b1, train_cfg.b2))

    # --- Dataset / Loader ---
    dataset = ImageMatrixDataset(
        file_d=io.file_d,
        file_c=io.file_c,
        img_size=model_cfg.img_size,
        val_mask=masks.val_mask,
        transform=ToTensor(),
    )
    loader = DataLoader(
        dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)

    gamma = float(train_cfg.gamma)
    k = float(train_cfg.kt)
    lambda_k = float(train_cfg.lambda_k)

    best_mse = np.inf
    patience = 0
    total = len(loader) * train_cfg.n_epochs
    pbar = tqdm(total=total)

    for epoch in range(train_cfg.n_epochs):
        for i, batch in enumerate(loader):
            real = batch["data"].to(device)   # (B,1,H,W)
            mask = batch["mask"].to(device)   # (B,1,H,W)
            label = batch["label"].to(device)
            label_oh = one_hot(label, model_cfg.ncls).to(device)

            # ----- Generator (Decoder) step -----
            optimizer_D.zero_grad()

            _, _, gen_vae = generator(real, label_oh)

            z = torch.randn(real.size(0), model_cfg.latent_dim, device=device)
            gen_img = generator.decoder(z, label_oh)

            real_in = torch.cat((real, mask), dim=1)
            fakeV_in = torch.cat((gen_vae, mask), dim=1)
            fakeG_in = torch.cat((gen_img, mask), dim=1)

            d_real_out = discriminator(real_in, label_oh)
            d_fakeV_out = discriminator(fakeV_in, label_oh)

            # Reconstruction loss on observed region
            rec_map = torch.abs(d_fakeV_out - d_real_out) * mask
            rec_loss = rec_map.sum() / mask.sum()

            # Adversarial (L1) loss
            d_fakeG_out = discriminator(fakeG_in, label_oh)
            l1_map1 = torch.abs(d_fakeG_out - gen_img) * mask

            d_fakeV2_out = discriminator(fakeV_in, label_oh)
            l1_map2 = torch.abs(d_fakeV2_out - gen_vae) * mask

            l1_loss = (l1_map1.sum() + l1_map2.sum()) / (2.0 * mask.sum())

            g_loss = 0.1 * rec_loss + l1_loss
            g_loss.backward()
            optimizer_D.step()

            # ----- Discriminator step -----
            optimizer_Dis.zero_grad()

            d_real_o = discriminator(real_in, label_oh)
            d_fake_o = discriminator(fakeG_in.detach(), label_oh)
            d_fakeV_o = discriminator(fakeV_in.detach(), label_oh)

            dr_map = torch.abs(d_real_o - real) * mask
            df_map = torch.abs(d_fake_o - gen_img.detach()) * mask
            dv_map = torch.abs(d_fakeV_o - gen_vae.detach()) * mask

            d_loss_real = dr_map.sum() / mask.sum()
            d_loss_fake = df_map.sum() / mask.sum()
            d_loss_v = dv_map.sum() / mask.sum()

            d_loss = d_loss_real - k * 0.5 * (d_loss_fake + d_loss_v)
            d_loss.backward()
            optimizer_Dis.step()

            # ----- Encoder step -----
            optimizer_E.zero_grad()

            mu2, lv2, gen_vae2 = generator(real, label_oh)
            fakeV2_in = torch.cat((gen_vae2, mask), dim=1)

            d_fakeV2 = discriminator(fakeV2_in, label_oh)
            d_real2 = discriminator(real_in, label_oh)

            rec2 = ((d_fakeV2 - d_real2) ** 2 * mask).sum() / mask.sum()
            kl2 = -0.5 * torch.sum(1 + lv2 - mu2.pow(2) -
                                   lv2.exp()) / mu2.numel()

            err_enc = kl2 + 5.0 * rec2
            err_enc.backward()
            optimizer_E.step()

            # ----- Update k -----
            diff = torch.mean(gamma * d_loss_real - 0.5 *
                              (d_loss_fake + d_loss_v))
            k = min(max(k + lambda_k * float(diff.item()), 0.0), 1.0)

            pbar.set_postfix({
                "epoch": epoch + 1,
                "batch": i + 1,
                "d_loss": float(d_loss.item()),
                "g_loss": float(g_loss.item()),
                # "k": k,
            })
            pbar.update(1)

        # ----- Epoch-end validation MSE on dropped entries -----
        with torch.no_grad():
            fake = generator.decoder(z_fixed, lbls_oh)  # (n_cells,1,H,W)
            fill = fake.detach().cpu().numpy()
            fill = flatten_images_to_matrix(fill)  # (n_feat,n_cells)

            idx = masks.obs_mask & (~masks.val_mask)

            mse = float(((fill - df_raw) ** 2)[idx].mean())

        min_delta = 1e-4
        if np.isfinite(mse) and mse < (best_mse - min_delta):
            best_mse = mse
            patience = 0

            torch.save(generator.encoder.state_dict(), ckpt.enc_best)
            torch.save(generator.decoder.state_dict(), ckpt.dec_best)
            torch.save(discriminator.state_dict(), ckpt.d_best)
        else:
            patience += 1

        if patience >= train_cfg.patience:
            print(
                f"Early stopping at epoch {epoch+1}, best val mse={best_mse:.6g}")
            break

    pbar.close()

    if not ckpt.dec_best.is_file():
        raise RuntimeError(
            "Training finished but no checkpoint was saved (val MSE never improved).")

    return ckpt

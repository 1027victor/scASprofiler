from __future__ import annotations

import os

import click

from .config import IOConfig, ImputeConfig, ModelConfig, TrainConfig
from .utils import set_global_seed,load_and_preprocess_data


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
@click.option(
    "--cuda-device",
    type=str,
    default=0,
    help="Set CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1'). Must be set BEFORE torch initializes.",
)
@click.option(
    "--deterministic/--no-deterministic",
    default=True,
    show_default=True,
    help="Try to make torch ops deterministic.",
)
@click.pass_context
def cli(ctx: click.Context, seed: int, cuda_device: str | None, deterministic: bool):
    # """scASP: VAE-GAN + KNN imputation (Click CLI)."""
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    set_global_seed(seed, deterministic=deterministic)
    ctx.ensure_object(dict)
    ctx.obj["seed"] = seed


def _io(file_d: str, file_c: str, outdir: str, name: str) -> IOConfig:
    return IOConfig(file_d=file_d, file_c=file_c, outdir=outdir, job_name=name)


def _model(img_size: int, channels: int, latent_dim: int, clusters: int) -> ModelConfig:
    return ModelConfig(img_size=img_size, channels=channels, latent_dim=latent_dim, ncls=clusters)


@cli.command("train")
@click.option("--data-Sj", "file_d", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--data-c", "file_c", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--outdir", type=click.Path(file_okay=False), default=".", show_default=True)
@click.option("--name", default="", show_default=True)
@click.option("--n-epochs", type=int, default=200, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--lr", type=float, default=2e-4, show_default=True)
@click.option("--b1", type=float, default=0.5, show_default=True)
@click.option("--b2", type=float, default=0.999, show_default=True)
@click.option("--gamma", type=float, default=0.95, show_default=True)
@click.option("--kt", type=float, default=0.0, show_default=True)
@click.option("--drop-prob", type=float, default=0.1, show_default=True)
@click.option("--patience", type=int, default=10, show_default=True)
@click.option("--channels", type=int, default=1, show_default=True)
@click.option("--latent-dim", type=int, default=100, show_default=True)
@click.option("--clusters", type=int, default=4, show_default=True)
@click.option("--overwrite", is_flag=True, help="Overwrite existing checkpoints if they already exist.")
@click.option("--detect-anomaly", is_flag=True, help="Enable torch autograd anomaly detection.")
@click.option(
    "--run-impute/--no-run-impute",
    default=False,
    show_default=True,
    help="Run imputation immediately after training.",
)
@click.option("--sim-size", type=int, default=200, show_default=True)
@click.option("-k", type=int, default=10, show_default=True)
@click.pass_context
def train_cmd(
    ctx: click.Context,
    file_d: str,
    file_c: str,
    outdir: str,
    name: str,
    n_epochs: int,
    batch_size: int,
    lr: float,
    b1: float,
    b2: float,
    gamma: float,
    kt: float,
    drop_prob: float,
    patience: int,
    # threthold: float,
    channels: int,
    latent_dim: int,
    clusters: int,
    overwrite: bool,
    detect_anomaly: bool,
    run_impute: bool,
    sim_size: int,
    k: int,
):
    # """Train VAE-GAN (and optionally run imputation)."""
    _, fig_h = load_and_preprocess_data(file_d)

    io = _io(file_d, file_c, outdir, name)
    model_cfg = _model(fig_h, channels, latent_dim, clusters)
    train_cfg = TrainConfig(
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        b1=b1,
        b2=b2,
        gamma=gamma,
        kt=kt,
        drop_prob=drop_prob,
        patience=patience,
        # threthold=threthold,
        detect_anomaly=detect_anomaly,
    )

    from .train import train

    ckpt = train(io=io, model_cfg=model_cfg, train_cfg=train_cfg,overwrite=overwrite)
    click.echo(f"Saved best decoder: {ckpt.dec_best}")

    if run_impute:
        from .impute import impute

        imp_cfg = ImputeConfig(sim_size=sim_size, k=k)
        out_path = impute(
            io=io,
            model_cfg=model_cfg,
            impute_cfg=imp_cfg,
            decoder_ckpt=str(ckpt.dec_best),
            n_epochs_for_name=n_epochs,
        )
        click.echo(f"Imputed matrix saved to: {out_path}")


@cli.command("impute")
@click.option("--data-Sj", "file_d", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--data-c", "file_c", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--outdir", type=click.Path(file_okay=False), default=".", show_default=True)
@click.option("--name", default="", show_default=True)
@click.option("--channels", type=int, default=1, show_default=True)
@click.option("--latent-dim", type=int, default=100, show_default=True)
@click.option("--clusters", type=int, default=4, show_default=True)
@click.option("--sim-size", type=int, default=200, show_default=True)
@click.option("-k", type=int, default=10, show_default=True)
@click.option(
    "--decoder-ckpt",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a decoder checkpoint. If omitted, uses the default derived from filenames/job-name.",
)
@click.option(
    "--n-epochs-for-name",
    type=int,
    default=200,
    show_default=True,
    help="Only used to reconstruct the default checkpoint filename when --decoder-ckpt is omitted.",
)
def impute_cmd(
    file_d: str,
    file_c: str,
    outdir: str,
    name: str,
    channels: int,
    latent_dim: int,
    clusters: int,
    sim_size: int,
    k: int,
    decoder_ckpt: str | None,
    n_epochs_for_name: int,
):
    # """Impute using a trained decoder + class-conditional KNN refinement."""
    io = _io(file_d, file_c, outdir, name)
    _, fig_h = load_and_preprocess_data(file_d)
    model_cfg = _model(fig_h, channels, latent_dim,clusters)
    # imp_cfg = ImputeConfig(sim_size=sim_size, knn_k=k)
    imp_cfg = ImputeConfig(sim_size=sim_size, k=k)

    from .impute import impute

    out_path = impute(
        io=io,
        model_cfg=model_cfg,
        impute_cfg=imp_cfg,
        decoder_ckpt=decoder_ckpt,
        n_epochs_for_name=n_epochs_for_name,
    )
    click.echo(f"Imputed matrix saved to: {out_path}")


if __name__ == "__main__":
    cli()

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    img_size: int = 100
    channels: int = 1
    latent_dim: int = 100
    ncls: int = 4


@dataclass(frozen=True)
class TrainConfig:
    n_epochs: int = 200
    batch_size: int = 8
    lr: float = 2e-4
    b1: float = 0.5
    b2: float = 0.999
    gamma: float = 0.95
    kt: float = 0.0
    drop_prob: float = 0.1
    patience: int = 10
    # threthold: float = 1e-2
    lambda_k: float = 1e-3
    detect_anomaly: bool = False


@dataclass(frozen=True)
class ImputeConfig:
    sim_size: int = 200
    k: int = 10


@dataclass(frozen=True)
class IOConfig:
    file_d: str
    file_c: str
    outdir: str = "."
    job_name: str = ""


==============
scASprofiler-impute CLI
==============

The scASprofiler-impute CLI implements a VAE-GAN framework for imputing highly sparse single-cell alternative splicing (AS) / junction count matrices. 
This command supports two modes: training and imputation. In training mode, scASprofiler learns a conditional generative model that captures cluster-specific 
splicing distributions, where each cell is associated with a discrete label (cluster / cell type). In imputation mode, it uses the trained decoder to generate 
in silico splicing profiles and then performs weighted KNN-style imputation (median over nearest synthetic neighbors) to fill in missing entries, while preserving observed values.

you can generate a complete single-cell splice junction counts matrix  by the command line like this:

.. code-block:: bash

    scASprofiler-impute  train   --data-Sj /out/filter_sj_counts.csv   --data-c onlyfillna_as_PRJEB15062_smart_seq2_label.txt   --outdir ./  --clusters 2   --n-epochs 1000 --batch-size 8   --drop-prob 0.1 --patience 10   --overwrite   --run-impute   --name scasp_drop_0.1   -k 10



By default, you will have one output file in the outdir,the output file contains all information for quantify, for
`scASprofiler-quantify`.

Options
=======

There are more parameters for setting (``scASprofiler-impute train --help`` always give the version 
you are using):

.. code-block:: html

    parameter settings

    Options:
        data_sj: path to the splicing junction matrix file (feature × cell; missing values as NaN), used as the model input for training/imputation.
        data_c: path to the cell label/cluster file (one label per cell, aligned to the columns of the SJ matrix), used for conditional generation.
        outdir: output directory for saving checkpoints and imputed results.
        name: user-defined job name used to prefix output files; if empty, a name is derived from input filenames.
        n_epochs: number of training epochs.
        batch_size: mini-batch size used during training.
        drop_prob: fraction of observed entries randomly masked to build a pseudo-validation set for early stopping and model selection.
        patience: early-stopping patience; training stops if validation MSE does not improve for this many epochs.
        threthold: convergence threshold parameter (reserved for convergence control; may be used to judge training stabilization depending on implementation).
        channels: number of input channels for the reshaped SJ “image” (typically 1 for a single matrix).
        latent_dim: dimensionality of the latent space in the VAE encoder/decoder.
        clusters: number of clusters/classes in the label file (one-hot conditioning dimension).
        overwrite: whether to overwrite existing checkpoints with the same job configuration.
        run_impute: whether to run imputation immediately after training finishes.
        no_run_impute: disable automatic post-training imputation.
        sim_size: number of synthetic samples generated per cluster during imputation (larger gives smoother KNN statistics but costs more time/memory).
        k: number of nearest synthetic neighbors used in mask-aware KNN imputation.
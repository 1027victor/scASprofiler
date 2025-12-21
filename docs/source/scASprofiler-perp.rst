==============
scASprofiler-perp CLI
==============

scASprofiler provides three CLI directly available in your Python path: 
``scASprofiler-perp``, ``scASprofiler-impute``, ``scASprofiler-quantify`` . 

This module implements the core functionality of **scASprofiler** for building
a high-quality single-cell splicing junction (SJ) count matrix. Starting from
STAR junction outputs, it performs junction preprocessing, gene annotation
mapping with a GTF file, intron grouping by splice sites (3' / 5'), and
multi-round quality control (QC) filtering. The final outputs include
group-aware SJ matrices, a junction metadata table, and a normalized/padded
matrix ready for downstream modeling.

you can generate a filitered single-cell splice junction counts matrix a list of Sj files by the command line like this:

.. code-block:: bash

  # for smart-seq
  scASprofiler-perp  run --sj-dir /SJ --gtf gencode.v46.annotation.gtf --outdir ./out --samples-ps 1 --sites-ps 20 --sites-thres 10 --samples-thres 1000

  # for droplet, e.g. 10x Genomics
  scASprofiler-perp  run --sj-dir /SJ --gtf gencode.v46.annotation.gtf --outdir ./out --samples-ps 1 --sites-ps 20 --sites-thres 10 --samples-thres 1000 --x10



By default, you will have three output files in the outdir: ``filter_sj_counts.csv``, 
``sj_meta.csv``,  and ``raw_sj_counts.csv``. The 
``filter_sj_counts.csv`` contains all information for imputation, e.g., for
`scASprofiler-impute`.

Options
=======

There are more parameters for setting (``scASprofiler-perp run --help`` always give the version 
you are using):

.. code-block:: html

    parameter settings

    Options:
        sj_dir: directory of STAR splicing junction files (e.g., SJ.out.tab); used as the input junction table for filtering and grouping.
        gtf: GTF annotation file path.
        outdir: output directory for all generated results.
        samples_ps: group-wise thresholding parameter; minimum number of observed (non-NaN) cells per junction within an intron group—junctions below this are set to missing (NaN).
        sites_ps: group-wise thresholding parameter; minimum total counts per cell within an intron group—cells below this are set to missing (NaN) within that group.
        sites_thres: site-level QC threshold; minimum number of expressing cells per junction (row-wise non-NaN count) to retain a junction.
        samples_thres: cell-level QC threshold; minimum number of expressing junctions per cell (column-wise non-NaN count) to retain a cell.
        use_ray: whether to enable Ray parallelism for group-wise threshold filtering (useful for large datasets).
        num_cpus: number of CPUs to allocate for Ray-based parallel computation.
        filter_unique_gene: whether to keep only junctions uniquely assigned to a single gene (reduces cross-gene ambiguity).
        keep_multi_gene: whether to retain junctions that map to multiple genes (less strict; may include ambiguous loci).
        use_multi: whether to include multi-mapped reads/junction counts if present in the input matrix (behavior depends on how the upstream counts were generated).
        plate: pipeline switch; whether to use the plate-based workflow (smart-seq2).
        x10: alias for the 10x (droplet-based) pipeline; if enabled, overrides the plate/tenx selection.

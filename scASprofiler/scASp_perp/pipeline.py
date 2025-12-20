import pandas as pd
# import numpy as np

from .sj_processing import process_splicing_junctions
from .annotations import add_gene_annotations
from .grouping import group_introns
from .thresholding import thres_filter
from .qc import qc_filter_sites, qc_filter_samples
# from .normalize import make_square_and_normalize


META11 = [
    "chromosome", "strand", "start", "end",
    "gene_id", "gene_id_start", "gene_id_end", "n_genes",
    "gene_name", "coordinates", "annotated"
]


def process_and_merge_splicing_data(
    sj_path: str,
    gtf_path: str,
    data_path: str,
    filter_unique_gene: bool = True,
) -> pd.DataFrame:

    # Process splicing junctions
    sj = process_splicing_junctions(sj_path)

    # Add gene annotations
    sj_final = add_gene_annotations(
        sj, gtf_path, filter_unique_gene=filter_unique_gene)
    sj_final = sj_final[
        ["chromosome", "start", "end", "strand", "annotated",
         "gene_id_start", "gene_id_end", "n_genes", "gene_name", "gene_id", "coordinates"]
    ].copy()

    # Load additional data and find intersections
    data = pd.read_csv(data_path, sep="\t", index_col=0)
    data.index = data.index.astype(str)

    intersection = set(sj_final["coordinates"].astype(
        str)).intersection(set(data.index))
    intersection_list = list(intersection)

    data = data.loc[intersection_list]
    merged_df = data.merge(sj_final, left_index=True,
                           right_on="coordinates", how="inner")
    merged_df = merged_df.reset_index()
    return merged_df

# run smaert-seq data


def run_filter_pipeline_plate(
    sj_path: str,
    gtf_path: str,
    data_path: str,
    samples_ps: int,
    sites_ps: int,
    sites_thres: int,
    samples_thres: int,
    # enable_plot: bool = True,
    use_ray: bool = False,
    num_cpus: int = 50,
    filter_unique_gene: bool = True,
):

    print("processing and merging splicing data...")
    df = process_and_merge_splicing_data(
        sj_path=sj_path,
        gtf_path=gtf_path,
        data_path=data_path,
        filter_unique_gene=filter_unique_gene,
    )
    print("done.")

    if "index" not in df.columns:
        raise ValueError(
            "Merged dataframe has no 'index' column. Check merge step (reset_index).")

    print("executing repeat and initial threshold filter...")
    three_df = group_introns(df, by="three_prime")
    five_df = group_introns(df, by="five_prime")

    three_df = thres_filter(three_df, samples_ps, sites_ps,
                            use_ray=use_ray, num_cpus=num_cpus)
    five_df = thres_filter(five_df, samples_ps, sites_ps,
                           use_ray=use_ray, num_cpus=num_cpus)
    print("done")

    print("executing sites quality filter by threshold")

    three_df = qc_filter_sites(three_df, sites_thres)
    five_df = qc_filter_sites(five_df, sites_thres)
    print("done.")

    print("keep the duplicated site starts and ends (re-group)...")

    three_df = group_introns(three_df, by="three_prime")
    five_df = group_introns(five_df, by="five_prime")
    print("done.")

    print("executing sample quality filter...")
    # sample filter needs the combined dataframe
    df_repeat = pd.concat([three_df, five_df], axis=0)

    df_sample_filtered = qc_filter_samples(df_repeat, samples_thres)
    print("done.")

    three_df = three_df[df_sample_filtered.columns]
    five_df = five_df[df_sample_filtered.columns]

    three_df = group_introns(three_df, by="three_prime").sort_values(
        by=["gene_name", "chromosome", "strand", "start", "end"]
    )
    five_df = group_introns(five_df, by="five_prime").sort_values(
        by=["gene_name", "chromosome", "strand", "start", "end"]
    )

    three_df = three_df.rename(columns={"coord.intron": "index"})
    five_df = five_df.rename(columns={"coord.intron": "index"})

    three_df.insert(0, "index", three_df.pop("index"))
    five_df.insert(0, "index", five_df.pop("index"))

    three_df = thres_filter(three_df, samples_ps, sites_ps,
                            use_ray=use_ray, num_cpus=num_cpus)
    five_df = thres_filter(five_df, samples_ps, sites_ps,
                           use_ray=use_ray, num_cpus=num_cpus)

    if "coord.intron" in three_df.columns:
        coord_intron_three = three_df.pop("coord.intron")
        three_df["coord.intron"] = coord_intron_three
    if "coord.intron" in five_df.columns:
        coord_intron_five = five_df.pop("coord.intron")
        five_df["coord.intron"] = coord_intron_five

    # executing the final repeat filter
    three_df = group_introns(three_df, by="three_prime").sort_values(
        by=["gene_name", "chromosome", "strand", "start", "end"]
    )
    five_df = group_introns(five_df, by="five_prime").sort_values(
        by=["gene_name", "chromosome", "strand", "start", "end"]
    )

    sj_formation = pd.concat([three_df, five_df], axis=0)[
        ["coord.intron", "annotated", "chromosome", "strand", "start", "end",
         "gene_id", "gene_name", "intron_group", "intron_group_size", "n_genes_per_intron_group"]
    ].copy()

    # new add
    sj_formation[["annotated", "start", "end"]] = sj_formation[[
        "annotated", "start", "end"]].apply(pd.to_numeric, errors="coerce").astype("Int64")

    drop_cols = [
        "chromosome", "strand", "start", "end",
        "gene_id", "gene_id_start", "gene_id_end", "n_genes",
        "gene_name", "coordinates",
        "intron_group_size", "n_genes_per_intron_group",
    ]
    three_df = three_df.drop(
        columns=[c for c in drop_cols if c in three_df.columns], errors="ignore")
    five_df = five_df.drop(
        columns=[c for c in drop_cols if c in five_df.columns], errors="ignore")

    # new add
    for df in [three_df, five_df]:
        if "annotated" in df.columns:
            df["annotated"] = pd.to_numeric(
                df["annotated"], errors="coerce").astype("Int64")

    merge_df = pd.concat([three_df, five_df], axis=0)

    must_last = [c for c in ["intron_group", "annotated",
                             "coord.intron"] if c in merge_df.columns]
    cell_cols = [c for c in merge_df.columns if c not in must_last]
    merge_df = merge_df[cell_cols + must_last].copy()

    if len(must_last) != 3:
        raise ValueError(
            f"Expected last 3 meta cols (intron_group, annotated, coord.intron), "
            f"but got: {must_last}. Check column dropping/order."
        )

    return sj_formation, merge_df


# run 10x data
def run_filter_pipeline_10x(
    sj_path: str,
    gtf_path: str,
    data_path: str,
    samples_ps: int,
    sites_ps: int,
    sites_thres: int,
    samples_thres: int,
    # enable_plot: bool = True,
    use_ray: bool = False,
    num_cpus: int = 50,
    filter_unique_gene: bool = True,
):

    print("processing and merging splicing data...")
    df = process_and_merge_splicing_data(
        sj_path=sj_path,
        gtf_path=gtf_path,
        data_path=data_path,
        filter_unique_gene=filter_unique_gene,
    )
    print("done.")

    if "index" not in df.columns:
        raise ValueError(
            "Merged dataframe has no 'index' column. Check merge step (reset_index).")

    print("executing repeat and initial threshold filter...")
    gene_df = group_introns(df, by="gene")

    gene_df = thres_filter(gene_df, samples_ps, sites_ps,
                           use_ray=use_ray, num_cpus=num_cpus)
    print("done")

    print("executing sites quality filter by threshold")

    gene_df = qc_filter_sites(gene_df, sites_thres)
    print("done.")

    print("keep the duplicated site starts and ends (re-group)...")

    gene_df = group_introns(gene_df, by="gene")
    print("done.")

    print("executing sample quality filter...")

    df_repeat = pd.concat([gene_df], axis=0)

    df_sample_filtered = qc_filter_samples(df_repeat, samples_thres)
    print("done.")

    gene_df = gene_df[df_sample_filtered.columns]

    gene_df = group_introns(gene_df, by="gene").sort_values(
        by=["gene_name", "chromosome", "strand", "start", "end"]
    )

    gene_df = gene_df.rename(columns={"coord.intron": "index"})

    gene_df.insert(0, "index", gene_df.pop("index"))

    gene_df = thres_filter(gene_df, samples_ps, sites_ps,
                           use_ray=use_ray, num_cpus=num_cpus)

    if "coord.intron" in gene_df.columns:
        coord_intron_three = gene_df.pop("coord.intron")
        gene_df["coord.intron"] = coord_intron_three

    gene_df = group_introns(gene_df, by="gene").sort_values(
        by=["gene_name", "chromosome", "strand", "start", "end"]
    )

    sj_formation = pd.concat([gene_df], axis=0)[
        ["coord.intron", "annotated", "chromosome", "strand", "start", "end",
         "gene_id", "gene_name", "intron_group", "intron_group_size", "n_genes_per_intron_group"]
    ].copy()

    sj_formation[["annotated", "start", "end"]] = sj_formation[[
        "annotated", "start", "end"]].apply(pd.to_numeric, errors="coerce").astype("Int64")

    drop_cols = [
        "chromosome", "strand", "start", "end",
        "gene_id", "gene_id_start", "gene_id_end", "n_genes",
        "gene_name", "coordinates",
        "intron_group_size", "n_genes_per_intron_group",
    ]
    gene_df = gene_df.drop(
        columns=[c for c in drop_cols if c in gene_df.columns], errors="ignore")

    for df in gene_df:
        if "annotated" in df.columns:
            df["annotated"] = pd.to_numeric(
                df["annotated"], errors="coerce").astype("Int64")

    merge_df = pd.concat([gene_df], axis=0)
    must_last = [c for c in ["intron_group", "annotated",
                             "coord.intron"] if c in merge_df.columns]
    cell_cols = [c for c in merge_df.columns if c not in must_last]
    merge_df = merge_df[cell_cols + must_last].copy()
    if len(must_last) != 3:
        raise ValueError(
            f"Expected last 3 meta cols (intron_group, annotated, coord.intron), "
            f"but got: {must_last}. Check column dropping/order."
        )
    return sj_formation, merge_df

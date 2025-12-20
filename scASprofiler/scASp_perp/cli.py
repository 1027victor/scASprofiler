import os
import click

@click.group()
def cli():
    """scASp: STAR splice junction pipeline."""
    pass



@cli.command("run")
@click.option("--sj-dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--gtf", "gtf_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--outdir", type=click.Path(file_okay=False), required=True)

@click.option("--samples-ps", type=int, default=1, show_default=True)
@click.option("--sites-ps", type=int, default=20, show_default=True)
@click.option("--sites-thres", type=int, default=10, show_default=True)
@click.option("--samples-thres", type=int, default=1000, show_default=True)


@click.option("--use-ray", is_flag=True)
@click.option("--num-cpus", type=int, default=5, show_default=True)
@click.option("--filter-unique-gene/--keep-multi-gene", default=True, show_default=True)
@click.option("--use-multi", is_flag=True)

@click.option("--plate", "is_plate", default=True, show_default=True,
              help="Choose pipeline: --plate uses run_filter_pipeline_plate; --tenx uses run_filter_pipeline_10x.")

@click.option("--x10", "x10_alias", is_flag=True,
              help="Alias for --tenx (10x pipeline). If set, overrides --plate/--tenx.")
def run_cmd(
    sj_dir, gtf_path, outdir,
    samples_ps, sites_ps, sites_thres, samples_thres,
    use_ray, num_cpus, filter_unique_gene, use_multi,
    is_plate, x10_alias
):
    # Delayed import: only load dependencies when actually running
    from .merge_matrix import discover_sj_files, build_row_merged_out_tab, build_junction_by_sample_matrix
    from .pipeline import run_filter_pipeline_plate, run_filter_pipeline_10x


    if x10_alias:
        is_plate = False

    os.makedirs(outdir, exist_ok=True)
    sj_files = discover_sj_files(sj_dir)

    merged_out_tab = os.path.join(outdir, "merged_sj.tab")
    matrix_csv = os.path.join(outdir, "raw_sj_counts.csv")

    click.echo("Generating input files from the sj directory...")
    build_row_merged_out_tab(sj_files, merged_out_tab)
    build_junction_by_sample_matrix(sj_files, matrix_csv, use_multi=use_multi)

    click.echo("Executing the main filtering pipeline:")

    if is_plate:
        click.echo("Mode: plate (run_filter_pipeline_plate)")
        sj_formation, merge_df = run_filter_pipeline_plate(
            sj_path=merged_out_tab,
            gtf_path=gtf_path,
            data_path=matrix_csv,
            samples_ps=samples_ps,
            sites_ps=sites_ps,
            sites_thres=sites_thres,
            samples_thres=samples_thres,
            use_ray=use_ray,
            num_cpus=num_cpus,
            filter_unique_gene=filter_unique_gene,
        )
    else:
        click.echo("Mode: 10x (run_filter_pipeline_10x)")
        sj_formation, merge_df = run_filter_pipeline_10x(
            sj_path=merged_out_tab,
            gtf_path=gtf_path,
            data_path=matrix_csv,
            samples_ps=samples_ps,
            sites_ps=sites_ps,
            sites_thres=sites_thres,
            samples_thres=samples_thres,
            use_ray=use_ray,
            num_cpus=num_cpus,
            filter_unique_gene=filter_unique_gene,
        )

    sj_formation.to_csv(os.path.join(outdir, "sj_meta.csv"), sep="\t", index=False)
    merge_df.to_csv(os.path.join(outdir, "filter_sj_counts.csv"), sep="\t", index=True)

    click.echo("Done.")

if __name__ == "__main__":
    cli()

import click
import os
import pandas as pd


def calculate_as_rate(sj_fill):
    """
    Calculate AS rate by group.

    Parameters:
    - sj_fill: DataFrame of splicing junctions(imputed).

    Returns:
    - as_rate: DataFrame containing AS rate by group.
    """
    value_columns = sj_fill.columns[3:]

    grouped = sj_fill.groupby('intron_group')[value_columns].transform('sum')

    percentage_cols = (sj_fill[value_columns].div(grouped.values))

    sj_fill[value_columns] = percentage_cols

    sj_fill.set_index('coord.intron', inplace=True)
    return sj_fill


@click.command()
@click.option(
    "--input-file", "input_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Input CSV file containing splice junction data."
)
@click.option(
    "--outdir", "output_dir",
    type=click.Path(exists=False, file_okay=False, writable=True),
    required=True,
    help="Output directory to save results."
)
def main(input_file, output_dir):
    """
    Read input, calculate AS ratio, and save output CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    sj_fill = pd.read_csv(input_file, index_col=0)
    as_rate = calculate_as_rate(sj_fill)

    output_file = os.path.join(output_dir, "as_ratio.csv")
    as_rate.to_csv(output_file)

    click.echo(
        f"AS ratio calculation complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()

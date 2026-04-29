# Fetch FTW field predictions for a bounding box and export per-year GeoParquet files.
#
# Usage:
#   python scripts/kuva/fetch_ftw_fields.py \
#     --bbox "xmin,ymin,xmax,ymax" \
#     --output-dir <dir> \
#     --name <prefix> \
#     --label <field|field_boundaries|non_field_background>
#
# Arguments:
#   --bbox        Bounding box in EPSG:4326 as "xmin,ymin,xmax,ymax" (required)
#   --output-dir  Directory to write output GeoParquet files, one per year (required)
#   --name        Prefix for output filenames, e.g. "france" → france_2024.parquet (default: fields)
#   --label       Prediction class to extract (default: field)
#
# Example — agricultural fields in a region of France:
#   python scripts/kuva/fetch_ftw_fields.py \
#     --bbox "-0.940450,47.843589,-0.774047,47.930975" \
#     --output-dir ./output \
#     --name france_fields \
#     --label field
#
# Output:
#   ./output/france_fields_2024.parquet
#   ./output/france_fields_2025.parquet

import time
import click
import duckdb
import geopandas as gpd
from pathlib import Path


DEFAULT_PARQUET_URL = (
    "s3://ftw/global-data/predictions/vectors/alpha/results/*.parquet"
)


def fetch_fields(
    con: duckdb.DuckDBPyConnection,
    parquet_url: str,
    xmin: float, ymin: float, xmax: float, ymax: float,
    label: str = "field",
) -> gpd.GeoDataFrame:
    q = f"""
        SELECT geometry AS geometry, time, label, bbox
        FROM read_parquet('{parquet_url}')
        WHERE
            label = '{label}'
            AND struct_extract(bbox, 'xmax') >= {xmin}
            AND struct_extract(bbox, 'xmin') <= {xmax}
            AND struct_extract(bbox, 'ymax') >= {ymin}
            AND struct_extract(bbox, 'ymin') <= {ymax}
    """
    df = con.execute(q).df()
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkb(df["geometry"].apply(bytes), crs="EPSG:4326"),
    )


@click.command()
@click.option("--bbox", required=True, type=str,
              help="Bounding box as 'xmin,ymin,xmax,ymax' in EPSG:4326.")
@click.option("--output-dir", required=True, type=click.Path(file_okay=False, writable=True, path_type=Path),
              help="Directory where the output GeoParquet files will be written.")
@click.option("--name", default="fields", show_default=True,
              help="Prefix for output file names (e.g. 'italy' produces 'italy_2024.parquet').")
@click.option("--label", default="field", show_default=True,
              type=click.Choice(["field", "field_boundaries", "non_field_background"], case_sensitive=True),
              help="Prediction label to filter on.")
@click.option("--parquet-url", default=DEFAULT_PARQUET_URL, show_default=True,
              help="S3 URL pattern for the source parquet files.")
@click.option("--s3-endpoint", default="data.source.coop", show_default=True,
              help="S3-compatible endpoint.")
def fetch_ftw_fields(bbox: str, output_dir: Path, name: str, label: str, parquet_url: str, s3_endpoint: str) -> None:
    """Fetch FTW field predictions for a bounding box and export per-year GeoParquet files."""
    t_start = time.perf_counter()

    try:
        xmin, ymin, xmax, ymax = [float(v) for v in bbox.split(",")]
    except ValueError:
        raise click.BadParameter("Expected format: 'xmin,ymin,xmax,ymax'", param_hint="--bbox")

    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"""
        SET s3_endpoint = '{s3_endpoint}';
        SET s3_url_style = 'path';
        SET s3_use_ssl = true;
        SET s3_region = 'us-west-2';
    """)

    click.echo(f"Fetching '{label}' features for bbox: {xmin},{ymin},{xmax},{ymax} ...")

    t_query = time.perf_counter()
    gdf = fetch_fields(con, parquet_url, xmin, ymin, xmax, ymax, label)
    click.echo(f"  Query completed in {time.perf_counter() - t_query:.1f}s — {len(gdf)} features found.")

    if gdf.empty:
        click.echo("No fields found for the given bounding box.")
        return

    for year, gdf_year in gdf.groupby(gdf["time"].dt.year):
        t_write = time.perf_counter()
        out_path = output_dir / f"{name}_{year}.parquet"
        gdf_year.to_parquet(out_path, index=False)
        click.echo(f"  Wrote {len(gdf_year)} features for {year} → {out_path} ({time.perf_counter() - t_write:.1f}s)")

    click.echo(f"Done in {time.perf_counter() - t_start:.1f}s total.")


if __name__ == "__main__":
    fetch_ftw_fields()
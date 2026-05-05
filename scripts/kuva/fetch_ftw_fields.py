# Fetch FTW field predictions for a bounding box or GeoJSON AOI and export per-year GeoParquet files.
#
# Usage:
#   # Option A: bounding box
#   python scripts/kuva/fetch_ftw_fields.py \
#     --bbox "xmin,ymin,xmax,ymax" \
#     --output-dir <dir> \
#     --name <prefix>
#
#   # Option B: GeoJSON file (parcels)
#   python scripts/kuva/fetch_ftw_fields.py \
#     --geojson parcels.geojson \
#     --output-dir <dir> \
#     --name <prefix>
#
# Arguments:
#   --bbox        Bounding box in EPSG:4326 as "xmin,ymin,xmax,ymax"
#   --geojson     Path to a GeoJSON file whose geometries define the AOI
#                 (only fields intersecting these geometries are returned)
#   --output-dir  Directory to write output GeoParquet files, one per year (required)
#   --name        Prefix for output filenames (default: fields)
#   --label       Prediction class to extract (default: field)
#
# When --geojson is used the script:
#   1. Reads the GeoJSON and computes its total bounding box.
#   2. Uses that bbox for the remote S3 query (coarse filter).
#   3. Spatially joins the results against the GeoJSON parcels (fine filter).
#
# Example:
#   python scripts/kuva/fetch_ftw_fields.py \
#     --geojson ./parcels.geojson \
#     --output-dir ./output \
#     --name france_fields \
#     --label field

import time
import click
import duckdb
import geopandas as gpd
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_PARQUET_URL = (
    "s3://ftw/global-data/predictions/vectors/alpha/results/*.parquet"
)


def _make_connection(s3_endpoint: str) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection configured for S3/httpfs (httpfs must already be installed)."""
    con = duckdb.connect()
    con.execute("LOAD httpfs;")
    con.execute(f"""
        SET s3_endpoint = '{s3_endpoint}';
        SET s3_url_style = 'path';
        SET s3_use_ssl = true;
        SET s3_region = 'us-west-2';
        SET http_timeout = 600000;
        SET http_retries = 5;
        SET http_retry_wait_ms = 2000;
        SET http_retry_backoff = 2;
    """)
    return con


def _query_batch(
    batch: list[str],
    xmin: float, ymin: float, xmax: float, ymax: float,
    label: str,
    s3_endpoint: str,
) -> pd.DataFrame:
    """Execute a bbox-filtered query over one batch of parquet files in a dedicated connection."""
    con = _make_connection(s3_endpoint)
    batch_str = ", ".join(f"'{f}'" for f in batch)
    q = f"""
        SELECT geometry AS geometry, time, label, bbox
        FROM read_parquet([{batch_str}])
        WHERE
            label = '{label}'
            AND struct_extract(bbox, 'xmax') >= {xmin}
            AND struct_extract(bbox, 'xmin') <= {xmax}
            AND struct_extract(bbox, 'ymax') >= {ymin}
            AND struct_extract(bbox, 'ymin') <= {ymax}
    """
    for attempt in range(3):
        try:
            return con.execute(q).df()
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return pd.DataFrame()  # unreachable


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
    if df.empty:
        return gpd.GeoDataFrame(columns=["geometry", "time", "label", "bbox"], geometry="geometry")
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkb(df["geometry"].apply(bytes), crs="EPSG:4326"),
    )


def list_parquet_files(con: duckdb.DuckDBPyConnection, parquet_url: str) -> list[str]:
    """List all parquet files matching the URL glob pattern."""
    try:
        return [row[0] for row in con.execute(f"SELECT * FROM glob('{parquet_url}')").fetchall()]
    except Exception:
        return [parquet_url]


def fetch_fields_batched(
    con: duckdb.DuckDBPyConnection,
    parquet_url: str,
    xmin: float, ymin: float, xmax: float, ymax: float,
    label: str = "field",
    batch_size: int = 20,
    workers: int = 8,
    s3_endpoint: str = "data.source.coop",
) -> gpd.GeoDataFrame:
    """Fetch fields in parallel batches to avoid HTTP timeouts on large file sets."""
    files = list_parquet_files(con, parquet_url)

    if len(files) <= 1:
        return fetch_fields(con, parquet_url, xmin, ymin, xmax, ymax, label)

    n_batches = (len(files) + batch_size - 1) // batch_size
    click.echo(
        f"  Processing {len(files)} parquet files in {n_batches} batches "
        f"of {batch_size} using {workers} workers..."
    )
    results: list[pd.DataFrame] = []
    batches = [files[i : i + batch_size] for i in range(0, len(files), batch_size)]

    with click.progressbar(length=n_batches, label="  Scanning", width=50,
                           item_show_func=lambda x: f" {x}" if x else "") as bar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_num = {
                executor.submit(_query_batch, batch, xmin, ymin, xmax, ymax, label, s3_endpoint): batch_num
                for batch_num, batch in enumerate(batches, start=1)
            }
            for future in as_completed(future_to_num):
                batch_num = future_to_num[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results.append(df)
                    bar.update(1, f"batch {batch_num}/{n_batches} ({len(df)} features)")
                except Exception as e:
                    bar.update(1)
                    click.echo(f"\n  Warning: batch {batch_num} failed: {e}", err=True)

    if not results:
        return gpd.GeoDataFrame(columns=["geometry", "time", "label", "bbox"], geometry="geometry")

    combined = pd.concat(results, ignore_index=True)
    return gpd.GeoDataFrame(
        combined,
        geometry=gpd.GeoSeries.from_wkb(combined["geometry"].apply(bytes), crs="EPSG:4326"),
    )


def clip_to_geojson(gdf: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only features from *gdf* that intersect any geometry in *aoi*.

    Uses a spatial join (inner, 'intersects') so that every returned field
    touched at least one parcel in the GeoJSON.  Duplicate field rows
    (when a field overlaps multiple parcels) are dropped.
    """
    if gdf.empty:
        return gdf

    # Ensure same CRS
    if aoi.crs is None:
        aoi = aoi.set_crs("EPSG:4326")
    elif aoi.crs != gdf.crs:
        aoi = aoi.to_crs(gdf.crs)

    # Spatial join — keep field rows that intersect any parcel
    joined = gpd.sjoin(gdf, aoi, how="inner", predicate="intersects")

    # Drop duplicates (a field may intersect multiple parcels) and the join index
    joined = joined.drop(columns=["index_right"], errors="ignore")
    # Deduplicate on hashable scalar columns only; bbox is a dict (unhashable)
    dedup_cols = [c for c in ["geometry", "time", "label"] if c in joined.columns]
    joined = joined.drop_duplicates(subset=dedup_cols)

    return joined.reset_index(drop=True)


@click.command()
@click.option("--bbox", required=False, default=None, type=str,
              help="Bounding box as 'xmin,ymin,xmax,ymax' in EPSG:4326.")
@click.option("--geojson", "geojson_path", required=False, default=None,
              type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
              help="Path to a GeoJSON file whose features define the AOI (parcels).")
@click.option("--output-dir", required=True,
              type=click.Path(file_okay=False, writable=True, path_type=Path),
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
@click.option("--batch-size", default=20, show_default=True, type=int,
              help="Number of parquet files to process per batch (reduce to avoid timeouts).")
@click.option("--workers", default=8, show_default=True, type=int,
              help="Number of parallel workers for batch queries.")
def fetch_ftw_fields(
    bbox: str | None,
    geojson_path: Path | None,
    output_dir: Path,
    name: str,
    label: str,
    parquet_url: str,
    s3_endpoint: str,
    batch_size: int,
    workers: int,
) -> None:
    """Fetch FTW field predictions for a bounding box or GeoJSON AOI and export per-year GeoParquet files."""

    # ── validate inputs ──────────────────────────────────────────────────
    if bbox is None and geojson_path is None:
        raise click.UsageError("Provide either --bbox or --geojson (or both).")

    t_start = time.perf_counter()

    # ── resolve AOI geometry and bounding box ────────────────────────────
    aoi_gdf: gpd.GeoDataFrame | None = None

    if geojson_path is not None:
        click.echo(f"Reading GeoJSON AOI from {geojson_path}")
        aoi_gdf = gpd.read_file(geojson_path)
        if aoi_gdf.crs is None:
            aoi_gdf = aoi_gdf.set_crs("EPSG:4326")
        else:
            aoi_gdf = aoi_gdf.to_crs("EPSG:4326")
        click.echo(f"  {len(aoi_gdf)} parcel(s) loaded.")

        # Derive bounding box from the GeoJSON extent
        gj_bounds = aoi_gdf.total_bounds  # [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = gj_bounds

        # If the user also passed --bbox, intersect the two extents
        if bbox is not None:
            try:
                bxmin, bymin, bxmax, bymax = [float(v) for v in bbox.split(",")]
            except ValueError:
                raise click.BadParameter("Expected format: 'xmin,ymin,xmax,ymax'", param_hint="--bbox")
            xmin = max(xmin, bxmin)
            ymin = max(ymin, bymin)
            xmax = min(xmax, bxmax)
            ymax = min(ymax, bymax)
    else:
        # bbox-only mode (original behaviour)
        try:
            xmin, ymin, xmax, ymax = [float(v) for v in bbox.split(",")]
        except ValueError:
            raise click.BadParameter("Expected format: 'xmin,ymin,xmax,ymax'", param_hint="--bbox")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── DuckDB / S3 setup ────────────────────────────────────────────────
    # Install extensions once (idempotent); worker threads will only LOAD
    _setup = duckdb.connect()
    _setup.execute("INSTALL httpfs; INSTALL spatial;")

    con = _make_connection(s3_endpoint)
    con.execute("LOAD spatial;")

    # ── coarse fetch (bbox) ──────────────────────────────────────────────
    click.echo(f"Fetching '{label}' features for bbox: {xmin},{ymin},{xmax},{ymax}")

    t_query = time.perf_counter()
    gdf = fetch_fields_batched(con, parquet_url, xmin, ymin, xmax, ymax, label, batch_size, workers, s3_endpoint)
    click.echo(f"  Query completed in {time.perf_counter() - t_query:.1f}s — {len(gdf)} features found.")

    if gdf.empty:
        click.echo("No fields found for the given AOI.")
        return

    # ── fine filter (intersect with GeoJSON parcels) ─────────────────────
    if aoi_gdf is not None:
        t_clip = time.perf_counter()
        pre_count = len(gdf)
        gdf = clip_to_geojson(gdf, aoi_gdf)
        click.echo(
            f"  Spatial intersection reduced {pre_count} → {len(gdf)} features "
            f"({time.perf_counter() - t_clip:.1f}s)"
        )
        if gdf.empty:
            click.echo("No fields intersect the GeoJSON parcels.")
            return

    # ── write per-year parquet ───────────────────────────────────────────
    for year, gdf_year in gdf.groupby(gdf["time"].dt.year):
        t_write = time.perf_counter()
        out_path = output_dir / f"{name}_{year}.parquet"
        gdf_year.to_parquet(out_path, index=False)
        click.echo(f"  Wrote {len(gdf_year)} features for {year} → {out_path} ({time.perf_counter() - t_write:.1f}s)")

    click.echo(f"Done in {time.perf_counter() - t_start:.1f}s total.")


if __name__ == "__main__":
    fetch_ftw_fields()
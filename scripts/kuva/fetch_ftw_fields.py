# Fetch FTW field predictions for a bounding box or GeoJSON AOI and export per-year GeoParquet files.
#
# The script can be used fully interactively (no flags needed) or driven by flags
# for scripted / non-interactive runs.  Any flag that is omitted will be asked
# at runtime via a prompted wizard.
#
# Usage:
#   # Fully interactive wizard
#   python scripts/kuva/fetch_ftw_fields.py
#
#   # Partially pre-filled
#   python scripts/kuva/fetch_ftw_fields.py --geojson parcels.geojson
#
#   # Fully non-interactive
#   python scripts/kuva/fetch_ftw_fields.py \
#     --geojson parcels.geojson \
#     --output-dir ./output \
#     --name <prefix>
#
# Optional flags (all values can be provided interactively instead):
#   --bbox        Bounding box in EPSG:4326 as "xmin,ymin,xmax,ymax"
#   --geojson     Path to a GeoJSON file whose geometries define the AOI
#   --output-dir  Directory to write output GeoParquet files, one per year
#   --years       Years to fetch: 2024, 2025, or both (default: both)
#   --name        Prefix for output filenames (default: fields)
#   --label       Prediction class to extract (default: field)

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import duckdb
import geopandas as gpd
import pandas as pd

DEFAULT_PARQUET_URL = "s3://ftw/global-data/predictions/vectors/alpha/results/*.parquet"


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
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
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
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    return pd.DataFrame()  # unreachable


def fetch_fields(
    con: duckdb.DuckDBPyConnection,
    parquet_url: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
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
        return gpd.GeoDataFrame(
            columns=["geometry", "time", "label", "bbox"], geometry="geometry"
        )
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.GeoSeries.from_wkb(df["geometry"].apply(bytes), crs="EPSG:4326"),
    )


def list_parquet_files(con: duckdb.DuckDBPyConnection, parquet_url: str) -> list[str]:
    """List all parquet files matching the URL glob pattern."""
    try:
        return [
            row[0]
            for row in con.execute(f"SELECT * FROM glob('{parquet_url}')").fetchall()
        ]
    except Exception:
        return [parquet_url]


def fetch_fields_batched(
    con: duckdb.DuckDBPyConnection,
    parquet_url: str,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
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

    with click.progressbar(
        length=n_batches,
        label="  Scanning",
        width=50,
        item_show_func=lambda x: f" {x}" if x else "",
    ) as bar:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_num = {
                executor.submit(
                    _query_batch, batch, xmin, ymin, xmax, ymax, label, s3_endpoint
                ): batch_num
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
        return gpd.GeoDataFrame(
            columns=["geometry", "time", "label", "bbox"], geometry="geometry"
        )

    combined = pd.concat(results, ignore_index=True)
    return gpd.GeoDataFrame(
        combined,
        geometry=gpd.GeoSeries.from_wkb(
            combined["geometry"].apply(bytes), crs="EPSG:4326"
        ),
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


def _prompt_geometry_selection(aoi):
    """Interactively prompt the user to choose geometries from a multi-feature GeoDataFrame.

    Returns
    -------
    tuple[str, int | None]
        ``('single', idx)``         — use only ``aoi.iloc[idx]``
        ``('all-combined', None)``   — use total_bounds / all geometries (single output)
        ``('all-individual', None)`` — one output set per geometry
    """
    click.echo(f"\nGeoJSON contains {len(aoi)} geometries:")
    for i in range(len(aoi)):
        row = aoi.iloc[i]
        name = None
        for col in ("name", "NAME", "id", "ID", "label", "LABEL"):
            if col in aoi.columns and row[col] is not None and str(row[col]).strip():
                name = str(row[col])
                break
        bounds = row.geometry.bounds
        label = f"{name} — " if name else ""
        click.echo(
            f"  {i}: {label}{bounds[0]:.4f},{bounds[1]:.4f},{bounds[2]:.4f},{bounds[3]:.4f}"
        )
    click.echo("  all-combined   : merged bbox of all geometries (single output)")
    click.echo("  all-individual : one output set per geometry")

    choices = [str(i) for i in range(len(aoi))] + ["all-combined", "all-individual"]
    selection = click.prompt(
        "\nSelect geometry index or mode",
        type=click.Choice(choices, case_sensitive=False),
        default="all-combined",
    )

    if selection == "all-combined":
        return "all-combined", None
    if selection == "all-individual":
        return "all-individual", None
    return "single", int(selection)


@click.command()
@click.option(
    "--bbox",
    required=False,
    default=None,
    type=str,
    help="Bounding box as 'xmin,ymin,xmax,ymax' in EPSG:4326. Prompted if omitted.",
)
@click.option(
    "--geojson",
    "geojson_path",
    required=False,
    default=None,
    type=click.Path(dir_okay=False, readable=True, path_type=Path),
    help="Path to a GeoJSON file whose features define the AOI. Prompted if omitted.",
)
@click.option(
    "--output-dir",
    required=False,
    default=None,
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    help="Directory for output GeoParquet files. Prompted if omitted.",
)
@click.option(
    "--name",
    default="fields",
    show_default=True,
    help="Prefix for output file names (e.g. 'italy' produces 'italy_2024.parquet').",
)
@click.option(
    "--label",
    default="field",
    show_default=True,
    type=click.Choice(
        ["field", "field_boundaries", "non_field_background"], case_sensitive=True
    ),
    help="Prediction label to filter on.",
)
@click.option(
    "--parquet-url",
    default=DEFAULT_PARQUET_URL,
    show_default=True,
    help="S3 URL pattern for the source parquet files.",
)
@click.option(
    "--s3-endpoint",
    default="data.source.coop",
    show_default=True,
    help="S3-compatible endpoint.",
)
@click.option(
    "--batch-size",
    default=20,
    show_default=True,
    type=int,
    help="Number of parquet files to process per batch (reduce to avoid timeouts).",
)
@click.option(
    "--workers",
    default=8,
    show_default=True,
    type=int,
    help="Number of parallel workers for batch queries.",
)
@click.option(
    "--years",
    default=None,
    type=click.Choice(["2024", "2025", "both"], case_sensitive=False),
    help="Years to fetch (2024, 2025, or both). Prompted if omitted.",
)
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
    years: str | None,
) -> None:
    """Fetch FTW field predictions for a bounding box or GeoJSON AOI and export per-year GeoParquet files.

    All inputs can be provided as flags or entered interactively when omitted.
    """

    # Interactive wizard — ask for any value not supplied as a CLI flag.
    click.echo("")
    click.echo("=" * 60)
    click.echo(" FTW fields fetcher")
    click.echo("=" * 60)

    # Step 1: input source
    if bbox is None and geojson_path is None:
        source = click.prompt(
            "\nInput source",
            type=click.Choice(["geojson", "bbox"], case_sensitive=False),
            default="geojson",
        )
        if source == "geojson":
            geojson_path = Path(
                click.prompt(
                    "GeoJSON file path",
                    type=click.Path(exists=True, dir_okay=False, readable=True),
                )
            )
        else:
            raw = click.prompt("Bounding box (xmin,ymin,xmax,ymax)")
            bbox = raw
    elif geojson_path is not None and not geojson_path.exists():
        raise click.BadParameter(
            f"File not found: {geojson_path}", param_hint="--geojson"
        )

    # Step 2: resolve geometries from GeoJSON
    aoi_gdf: gpd.GeoDataFrame | None = None
    mode = None
    geo_suffixes: list[str | None] = [None]

    if geojson_path is not None:
        click.echo(f"\nReading GeoJSON AOI from {geojson_path}")
        aoi_gdf = gpd.read_file(geojson_path)
        if aoi_gdf.crs is None:
            aoi_gdf = aoi_gdf.set_crs("EPSG:4326")
        else:
            aoi_gdf = aoi_gdf.to_crs("EPSG:4326")
        click.echo(f"  {len(aoi_gdf)} parcel(s) loaded.")

        if len(aoi_gdf) > 1:
            mode, sel_idx = _prompt_geometry_selection(aoi_gdf)
        else:
            mode, sel_idx = "single", 0

        if mode == "all-individual":
            geo_suffixes = [str(i) for i in range(len(aoi_gdf))]
            aoi_slices = [aoi_gdf.iloc[[i]] for i in range(len(aoi_gdf))]
        else:
            geo_suffixes = [None]
            aoi_slices = [aoi_gdf.iloc[[sel_idx]] if mode == "single" else aoi_gdf]

        # bbox(es) from selected geometry/geometries
        def _clip(bounds):
            xn, yn, xx, yx = bounds
            if bbox is not None:
                try:
                    bxmin, bymin, bxmax, bymax = [float(v) for v in bbox.split(",")]
                except ValueError:
                    raise click.BadParameter(
                        "Expected 'xmin,ymin,xmax,ymax'", param_hint="--bbox"
                    )
                return [max(bxmin, xn), max(bymin, yn), min(bxmax, xx), min(bymax, yx)]
            return [xn, yn, xx, yx]

        if mode == "all-individual":
            bboxes = [
                _clip(aoi_gdf.iloc[i].geometry.bounds) for i in range(len(aoi_gdf))
            ]
        else:
            bboxes = [_clip(aoi_slices[0].total_bounds)]
    else:
        # bbox-only mode
        try:
            bxmin, bymin, bxmax, bymax = [float(v) for v in bbox.split(",")]
        except ValueError:
            raise click.BadParameter(
                "Expected 'xmin,ymin,xmax,ymax'", param_hint="--bbox"
            )
        bboxes = [[bxmin, bymin, bxmax, bymax]]
        aoi_slices = [None]

    # Step 3: years to fetch
    if years is None:
        years = click.prompt(
            "\nYears to fetch (only 2024 and 2025 are available)",
            type=click.Choice(["2024", "2025", "both"], case_sensitive=False),
            default="both",
        )
    selected_years = {2024, 2025} if years == "both" else {int(years)}

    # Step 4: output directory
    if output_dir is None:
        stem = geojson_path.stem if geojson_path else "ftw_fields"
        output_dir = Path(
            click.prompt("\nOutput directory", default=f"./{stem}_fields")
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # ── DuckDB / S3 setup ────────────────────────────────────────────────
    # Install extensions once (idempotent); worker threads will only LOAD
    _setup = duckdb.connect()
    _setup.execute("INSTALL httpfs; INSTALL spatial;")

    con = _make_connection(s3_endpoint)
    con.execute("LOAD spatial;")

    # ── fetch loop (one iteration per geometry in all-individual, else one) ──
    for geom_idx, (bbox_i, aoi_slice) in enumerate(zip(bboxes, aoi_slices)):
        xmin, ymin, xmax, ymax = bbox_i
        suffix = geo_suffixes[geom_idx]
        if len(bboxes) > 1:
            click.echo(f"\n--- Geometry {geom_idx} ({geom_idx + 1}/{len(bboxes)}) ---")

        click.echo(f"Fetching '{label}' features for bbox: {xmin},{ymin},{xmax},{ymax}")

        t_query = time.perf_counter()
        gdf = fetch_fields_batched(
            con,
            parquet_url,
            xmin,
            ymin,
            xmax,
            ymax,
            label,
            batch_size,
            workers,
            s3_endpoint,
        )
        click.echo(
            f"  Query completed in {time.perf_counter() - t_query:.1f}s — {len(gdf)} features found."
        )

        if gdf.empty:
            click.echo("  No fields found for this AOI.")
            continue

        if aoi_slice is not None:
            t_clip = time.perf_counter()
            pre_count = len(gdf)
            gdf = clip_to_geojson(gdf, aoi_slice)
            click.echo(
                f"  Spatial intersection reduced {pre_count} → {len(gdf)} features "
                f"({time.perf_counter() - t_clip:.1f}s)"
            )
            if gdf.empty:
                click.echo("  No fields intersect the selected geometry.")
                continue

        for year, gdf_year in gdf.groupby(gdf["time"].dt.year):
            if year not in selected_years:
                continue
            t_write = time.perf_counter()
            file_name = (
                f"{name}_{suffix}_{year}.parquet"
                if suffix is not None
                else f"{name}_{year}.parquet"
            )
            out_path = output_dir / file_name
            gdf_year.to_parquet(out_path, index=False)
            click.echo(
                f"  Wrote {len(gdf_year)} features for {year} → {out_path} "
                f"({time.perf_counter() - t_write:.1f}s)"
            )

    click.echo(f"\nDone in {time.perf_counter() - t_start:.1f}s total.")


if __name__ == "__main__":
    fetch_ftw_fields()

# Download an 8-band planting+harvest cube from the FTW global Zarr features dataset.
#
# Usage:
#   # Option A: explicit bounding box
#   python scripts/kuva/download_ftw_cube.py \
#     --bbox "xmin,ymin,xmax,ymax" \
#     --year 2025 \
#     --output ./ftw_cube.tif
#
#   # Option B: derive bbox from a GeoJSON file
#   python scripts/kuva/download_ftw_cube.py \
#     --geojson parcels.geojson \
#     --year 2025 \
#     --output ./ftw_cube.tif
#
# Arguments:
#   --bbox      Bounding box in EPSG:4326 as "xmin,ymin,xmax,ymax"
#   --geojson   Path to a GeoJSON file whose total extent is used as the bbox
#               (--bbox and --geojson can be combined; their extents are intersected)
#   --year      Year to download, e.g. 2025 (required)
#   --output    Path to write the output GeoTIFF (default: ./ftw_cube.tif)
#   --verbose   Enable verbose logging
#   --threads   Number of threads for parallel chunk download (default: os.cpu_count())
#
# Bands (in order): planting B04, B03, B02, B08 then harvest B04, B03, B02, B08
#
# Examples:
#   python scripts/kuva/download_ftw_cube.py \
#     --bbox "-65.983887,-35.137879,-62.094727,-29.113775" \
#     --year 2025 \
#     --output ./output/argentina_cube.tif \
#     --verbose
#
#   python scripts/kuva/download_ftw_cube.py \
#     --geojson ./parcels.geojson \
#     --year 2025 \
#     --output ./output/argentina_cube.tif

import datetime
import logging
import os
import time
from pathlib import Path

import click
import dask
import dask.callbacks
import geopandas as gpd
import rasterio
import rasterix
import s3fs
import tqdm
import xarray as xr

logger = logging.getLogger(__name__)

URL_ZARR_FEATURES = (
    "s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/features/zarr/alpha/global.zarr"
)

ALL_BANDS = [
    "s2med_planting:B04",
    "s2med_planting:B03",
    "s2med_planting:B02",
    "s2med_planting:B08",
    "s2med_harvest:B04",
    "s2med_harvest:B03",
    "s2med_harvest:B02",
    "s2med_harvest:B08",
]


class _ProgressCallback(dask.callbacks.Callback):
    """Dask callback showing chunk count and estimated download speed via tqdm."""

    def __init__(self, total_bytes: int) -> None:
        self._total_bytes = total_bytes
        self._n_tasks = 0
        self._t0: float = 0.0
        self._bar: tqdm.tqdm | None = None

    def _start_state(self, dsk, state) -> None:
        self._n_tasks = sum(len(state[k]) for k in ("ready", "waiting", "running", "finished"))
        self._t0 = time.perf_counter()
        self._bar = tqdm.tqdm(
            total=self._n_tasks,
            unit="chunk",
            desc="  Downloading",
            ncols=90,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} tasks  {postfix}",
        )

    def _posttask(self, key, result, dsk, state, worker_id) -> None:
        if self._bar is None:
            return
        elapsed = time.perf_counter() - self._t0
        done = self._bar.n + 1
        bytes_done = self._total_bytes * done / self._n_tasks if self._n_tasks else 0
        speed_mb = bytes_done / elapsed / 1e6 if elapsed > 0 else 0
        self._bar.set_postfix_str(f"{speed_mb:.1f} MB/s", refresh=False)
        self._bar.update(1)

    def _finish(self, dsk, state, errored) -> None:
        if self._bar is not None:
            elapsed = time.perf_counter() - self._t0
            avg_mb = self._total_bytes / elapsed / 1e6 if elapsed > 0 else 0
            self._bar.set_postfix_str(f"avg {avg_mb:.1f} MB/s", refresh=True)
            self._bar.close()


def _parse_bbox(ctx, param, value):
    if value is None:
        return None
    try:
        parts = [float(v) for v in value.split(",")]
        if len(parts) != 4:
            raise ValueError
        return parts
    except ValueError:
        raise click.BadParameter("expected 'xmin,ymin,xmax,ymax' as four floats")


@click.command()
@click.option(
    "--bbox",
    type=str,
    required=False,
    default=None,
    help="Bounding box as 'xmin,ymin,xmax,ymax' (EPSG:4326). Mutually exclusive with --geojson.",
    callback=_parse_bbox,
)
@click.option(
    "--geojson",
    "geojson_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=False,
    default=None,
    help="GeoJSON file whose total extent is used as the bounding box.",
)
@click.option(
    "--year",
    type=click.IntRange(min=2024, max=datetime.date.today().year),
    required=True,
    help="Year to download (e.g. 2025).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    default="./ftw_cube.tif",
    show_default=True,
    help="Path to write the output GeoTIFF.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
@click.option(
    "--threads",
    "-t",
    type=click.IntRange(min=1),
    default=None,
    show_default=True,
    help="Number of threads for parallel chunk download. Defaults to os.cpu_count().",
)
@click.option(
    "--s3-connections",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
    help="Number of parallel S3 connections.",
)
def main(bbox, geojson_path, year, output, verbose, threads, s3_connections):
    """Download an 8-band planting+harvest cube from the FTW global Zarr features dataset.

    Bands (in order): planting B04, B03, B02, B08 then harvest B04, B03, B02, B08.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    if bbox is None and geojson_path is None:
        raise click.UsageError("Provide either --bbox or --geojson.")

    if geojson_path is not None:
        aoi = gpd.read_file(geojson_path)
        if aoi.crs is None:
            aoi = aoi.set_crs("EPSG:4326")
        else:
            aoi = aoi.to_crs("EPSG:4326")
        gj_xmin, gj_ymin, gj_xmax, gj_ymax = aoi.total_bounds
        if bbox is not None:
            # Intersect GeoJSON extent with explicit --bbox
            xmin = max(bbox[0], gj_xmin)
            ymin = max(bbox[1], gj_ymin)
            xmax = min(bbox[2], gj_xmax)
            ymax = min(bbox[3], gj_ymax)
        else:
            xmin, ymin, xmax, ymax = gj_xmin, gj_ymin, gj_xmax, gj_ymax
        click.echo(f"Derived bbox from GeoJSON: {xmin},{ymin},{xmax},{ymax}")
    else:
        xmin, ymin, xmax, ymax = bbox

    bbox = [xmin, ymin, xmax, ymax]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    t_total = time.perf_counter()

    t0 = time.perf_counter()
    n_threads = threads or os.cpu_count() or 16 
    click.echo("Opening FTW global Zarr features store...")
    fs = s3fs.S3FileSystem(anon=True, config_kwargs={"max_pool_connections": s3_connections})
    store = s3fs.S3Map(root=URL_ZARR_FEATURES, s3=fs)
    features = xr.open_zarr(store, consolidated=True).pipe(rasterix.assign_index)
    click.echo(f"  Store opened in {time.perf_counter() - t0:.1f}s")

    native_chunks = dict(zip(features["variables"].dims, features["variables"].chunks))
    click.echo(f"  Native Zarr chunk sizes: { {k: v[0] for k, v in native_chunks.items()} }")

    click.echo(f"Slicing to bbox={bbox} year={year} ...")
    data = features["variables"].sel(
        band=ALL_BANDS,
        time=str(year),
        y=slice(ymax, ymin),
        x=slice(xmin, xmax),
    ).squeeze("time")

    click.echo(f"  Output chunks: {data.data.npartitions} (dask will schedule more tasks internally for slicing/squeezing)")

    t0 = time.perf_counter()
    click.echo(f"Loading data into memory (threads={n_threads}, s3_connections={s3_connections})...")
    with _ProgressCallback(data.nbytes), dask.config.set(scheduler="threads", num_workers=n_threads):
        data = data.compute()

    click.echo(f"  Data loaded in {time.perf_counter() - t0:.1f}s — shape: {data.shape}")

    arr = data.values  # (8, H, W)
    ys = data.y.values
    xs = data.x.values

    height, width = arr.shape[1], arr.shape[2]
    transform = rasterio.transform.from_bounds(
        west=float(xs.min()),
        south=float(ys.min()),
        east=float(xs.max()),
        north=float(ys.max()),
        width=width,
        height=height,
    )

    t0 = time.perf_counter()
    click.echo(f"Writing GeoTIFF to {output} ...")
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=len(ALL_BANDS),
        dtype=arr.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for i, band_data in enumerate(arr, start=1):
            dst.write(band_data, i)
        dst.update_tags(
            bands=",".join(ALL_BANDS),
            year=str(year),
            bbox=",".join(str(v) for v in bbox),
        )
    click.echo(f"  GeoTIFF written in {time.perf_counter() - t0:.1f}s")
    click.echo(f"Done. Output: {output} (total: {time.perf_counter() - t_total:.1f}s)")


if __name__ == "__main__":
    main()

# Download an 8-band planting+harvest cube from the FTW global Zarr features dataset.
#
# The script can be used fully interactively (no flags needed) or driven by flags
# for scripted / non-interactive runs.  Any flag that is omitted will be asked
# at runtime via a prompted wizard.
#
# Usage:
#   # Fully interactive wizard
#   python scripts/kuva/download_ftw_cube.py
#
#   # Partially pre-filled (remaining values are still prompted)
#   python scripts/kuva/download_ftw_cube.py --geojson parcels.geojson
#
#   # Fully non-interactive
#   python scripts/kuva/download_ftw_cube.py \
#     --geojson parcels.geojson \
#     --year 2025 \
#     --output ./ftw_cube.tif
#
# Optional flags (all values can be provided interactively instead):
#   --bbox      Bounding box in EPSG:4326 as "xmin,ymin,xmax,ymax"
#   --geojson   Path to a GeoJSON file (bbox is derived from chosen geometry)
#   --year      Year to download, e.g. 2025
#   --output    Path to write the output GeoTIFF
#   --verbose   Enable verbose logging
#   --threads   Number of threads for parallel chunk download (default: os.cpu_count())
#
# Bands (in order): planting B04, B03, B02, B08 then harvest B04, B03, B02, B08

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

URL_ZARR_FEATURES = "s3://us-west-2.opendata.source.coop/tge-labs/ftw-global-data/features/zarr/alpha/global.zarr"

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
        self._n_tasks = sum(
            len(state[k]) for k in ("ready", "waiting", "running", "finished")
        )
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


def _prompt_geometry_selection(aoi):
    """Interactively prompt the user to choose geometries from a multi-feature GeoDataFrame.

    Returns
    -------
    tuple[str, int | None]
        ``('single', idx)``        — use the bbox of ``aoi.iloc[idx]``
        ``('all-combined', None)``  — use ``total_bounds`` (single output file)
        ``('all-individual', None)``— one output file per geometry
    """
    click.echo(f"\nGeoJSON contains {len(aoi)} geometries:")
    for i in range(len(aoi)):
        row = aoi.iloc[i]
        name = None
        for col in ("name", "NAME", "id", "ID", "label", "LABEL"):
            if col in aoi.columns and row[col] is not None and str(row[col]).strip():
                name = str(row[col])
                break
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
        label = f"{name} — " if name else ""
        click.echo(
            f"  {i}: {label}{bounds[0]:.4f},{bounds[1]:.4f},{bounds[2]:.4f},{bounds[3]:.4f}"
        )
    click.echo("  all-combined   : merged bbox of all geometries (single output file)")
    click.echo("  all-individual : one output file per geometry")

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


def _download_and_write(features, bbox, year, output_path, n_threads):
    """Slice, download and write a single-bbox cube to *output_path* (GeoTIFF)."""
    xmin, ymin, xmax, ymax = bbox

    click.echo(f"Slicing to bbox={bbox} year={year} ...")
    data = (
        features["variables"]
        .sel(band=ALL_BANDS, time=str(year), y=slice(ymax, ymin), x=slice(xmin, xmax))
        .squeeze("time")
    )

    click.echo(
        f"  Output chunks: {data.data.npartitions}"
        " (dask will schedule more tasks internally for slicing/squeezing)"
    )

    t0 = time.perf_counter()
    click.echo(f"Loading data into memory (threads={n_threads})...")
    with (
        _ProgressCallback(data.nbytes),
        dask.config.set(scheduler="threads", num_workers=n_threads),
    ):
        data = data.compute()
    click.echo(
        f"  Data loaded in {time.perf_counter() - t0:.1f}s — shape: {data.shape}"
    )

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
    click.echo(f"Writing GeoTIFF to {output_path} ...")
    with rasterio.open(
        output_path,
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
    click.echo(f"  Output: {output_path}")


@click.command()
@click.option(
    "--bbox",
    type=str,
    required=False,
    default=None,
    help="Bounding box as 'xmin,ymin,xmax,ymax' (EPSG:4326). Prompted if omitted.",
    callback=_parse_bbox,
)
@click.option(
    "--geojson",
    "geojson_path",
    type=click.Path(dir_okay=False, readable=True),
    required=False,
    default=None,
    help="GeoJSON file. Prompted if omitted (and --bbox not given).",
)
@click.option(
    "--year",
    type=click.Choice(["2024", "2025"]),
    required=False,
    default=None,
    help="Year to download (2024 or 2025). Prompted if omitted.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Path to write the output GeoTIFF. Prompted if omitted.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging."
)
@click.option(
    "--threads",
    "-t",
    type=click.IntRange(min=1),
    default=None,
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

    All inputs can be provided as flags or entered interactively when omitted.
    Bands (in order): planting B04, B03, B02, B08 then harvest B04, B03, B02, B08.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Interactive wizard — ask for any value that was not supplied as a
    # CLI flag.

    click.echo("")
    click.echo("=" * 60)
    click.echo(" FTW global cube downloader")
    click.echo("=" * 60)

    # Step 1: input source
    if bbox is None and geojson_path is None:
        source = click.prompt(
            "\nInput source",
            type=click.Choice(["geojson", "bbox"], case_sensitive=False),
            default="geojson",
        )
        if source == "geojson":
            geojson_path = click.prompt(
                "GeoJSON file path",
                type=click.Path(exists=True, dir_okay=False, readable=True),
            )
        else:
            raw = click.prompt("Bounding box (xmin,ymin,xmax,ymax)")
            bbox = _parse_bbox(None, None, raw)
    elif geojson_path is not None and not Path(geojson_path).exists():
        raise click.BadParameter(
            f"File not found: {geojson_path}", param_hint="--geojson"
        )

    # Step 2: resolve geometries from GeoJSON
    if geojson_path is not None:
        aoi = gpd.read_file(geojson_path)
        if aoi.crs is None:
            aoi = aoi.set_crs("EPSG:4326")
        else:
            aoi = aoi.to_crs("EPSG:4326")

        if len(aoi) > 1:
            mode, sel_idx = _prompt_geometry_selection(aoi)
        else:
            mode, sel_idx = "single", 0

        def _clip(bounds):
            xn, yn, xx, yx = bounds
            if bbox is not None:
                return [
                    max(bbox[0], xn),
                    max(bbox[1], yn),
                    min(bbox[2], xx),
                    min(bbox[3], yx),
                ]
            return [xn, yn, xx, yx]

        if mode == "all-individual":
            bboxes = [_clip(aoi.iloc[i].geometry.bounds) for i in range(len(aoi))]
            # output path will be asked per-geometry below
            geo_suffixes = [str(i) for i in range(len(aoi))]
        else:
            bounds = (
                aoi.iloc[sel_idx].geometry.bounds
                if mode == "single"
                else aoi.total_bounds
            )
            bboxes = [_clip(bounds)]
            geo_suffixes = [None]
            click.echo(f"\nDerived bbox from GeoJSON: {bboxes[0]}")
    else:
        bboxes = [list(bbox)]
        geo_suffixes = [None]
        mode = None

    # Step 3: year
    if year is None:
        year = click.prompt(
            "\nYear to download (only 2024 and 2025 are available)",
            type=click.Choice(["2024", "2025"]),
            default="2025",
        )
    year = int(year)

    # Step 4: output path(s)
    if geojson_path and mode == "all-individual":
        if output is None:
            stem = Path(geojson_path).stem
            output = click.prompt("\nOutput folder", default=f"./{stem}_{year}")
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(geojson_path).stem
        outputs = [out_dir / f"{stem}_{s}_{year}.tif" for s in geo_suffixes]
        click.echo("\nOutput files:")
        for p in outputs:
            click.echo(f"  {p}")
    else:
        if output is None:
            stem = Path(geojson_path).stem if geojson_path else "ftw_cube"
            default_output = f"./{stem}_{year}.tif"
            output = click.prompt("\nOutput path", default=default_output)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        outputs = [output]

    # Step 5: threads
    if threads is None:
        threads = click.prompt(
            "\nDownload threads",
            type=click.IntRange(min=1),
            default=os.cpu_count() or 16,
        )

    t_total = time.perf_counter()

    t0 = time.perf_counter()
    n_threads = threads
    click.echo("Opening FTW global Zarr features store...")
    fs = s3fs.S3FileSystem(
        anon=True, config_kwargs={"max_pool_connections": s3_connections}
    )
    store = s3fs.S3Map(root=URL_ZARR_FEATURES, s3=fs)
    features = xr.open_zarr(store, consolidated=True).pipe(rasterix.assign_index)
    click.echo(f"  Store opened in {time.perf_counter() - t0:.1f}s")

    native_chunks = dict(zip(features["variables"].dims, features["variables"].chunks))
    click.echo(
        f"  Native Zarr chunk sizes: { {k: v[0] for k, v in native_chunks.items()} }"
    )

    for i, (bbox_i, output_i) in enumerate(zip(bboxes, outputs)):
        if len(bboxes) > 1:
            click.echo(f"\n--- Geometry {i} ({i + 1}/{len(bboxes)}) ---")
        _download_and_write(features, bbox_i, year, output_i, n_threads)

    click.echo(f"\nAll done. (total: {time.perf_counter() - t_total:.1f}s)")


if __name__ == "__main__":
    main()

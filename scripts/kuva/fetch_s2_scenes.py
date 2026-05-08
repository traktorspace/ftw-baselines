# Fetch Sentinel-2 planting/harvest scene pairs for a given area of interest.
#
# Usage:
#   # Option A: explicit bounding box — scene selection only
#   python scripts/kuva/fetch_s2_scenes.py \
#     --bbox "xmin,ymin,xmax,ymax" \
#     --year 2024
#
#   # Option B: derive bbox from a GeoJSON file + download the input tif
#   python scripts/kuva/fetch_s2_scenes.py \
#     --geojson parcels.geojson \
#     --year 2024 \
#     --create-input \
#     --output ./ftw_input.tif
#
# Arguments:
#   --bbox            Bounding box in EPSG:4326 as "xmin,ymin,xmax,ymax"
#   --geojson         Path to a GeoJSON file whose total extent is used as the bbox
#                     (--bbox and --geojson can be combined; their extents are intersected)
#   --year            Year to target, e.g. 2024 (required)
#   --output          Path to write the output GeoTIFF when --create-input is set
#                     (default: ./ftw_input.tif)
#   --create-input    Download the 8-band input tif (win_a + win_b merged)
#   --stac-host       STAC backend: 'mspc' (default) or 'earthsearch'
#   --s2-collection   Sentinel-2 collection, only relevant for EarthSearch (default: 'c1')
#   --cloud-cover-max Maximum cloud cover % (default: 10)
#   --buffer-days     Search window around crop-calendar dates in days (default: 60)
#   --nodata-max      Maximum nodata % per scene (default: 30)
#   --timeout         Seconds before create-input is aborted (default: 120)
#   --verbose / -v    Enable verbose logging
#
# Examples:
#   python scripts/kuva/fetch_s2_scenes.py \
#     --bbox "-65.983887,-35.137879,-62.094727,-29.113775" \
#     --year 2024 \
#     --verbose
#
#   python scripts/kuva/fetch_s2_scenes.py \
#     --geojson ./syngenta_1.geojson \
#     --year 2024 \
#     --create-input \
#     --output ./output/syngenta1_input.tif \
#     --verbose

import datetime
import logging
import signal
import sys
from pathlib import Path

import click
import dask
import geopandas as gpd

from ftw_tools.download.download_img import create_input, scene_selection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timeout helpers (POSIX / Linux only)
# ---------------------------------------------------------------------------


class _DownloadTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _DownloadTimeout("Download exceeded the allotted time")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--bbox",
    type=str,
    required=False,
    default=None,
    callback=_parse_bbox,
    help="Bounding box as 'xmin,ymin,xmax,ymax' (EPSG:4326).",
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
    type=click.IntRange(min=2017, max=datetime.date.today().year),
    required=True,
    help="Target year for scene selection (e.g. 2024).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    default="./ftw_input.tif",
    show_default=True,
    help="Output GeoTIFF path (only used with --create-input).",
)
@click.option(
    "--create-input",
    "do_create_input",
    is_flag=True,
    default=False,
    help="Download and merge the two scenes into a single 8-band GeoTIFF.",
)
@click.option(
    "--stac-host",
    type=click.Choice(["mspc", "earthsearch"], case_sensitive=False),
    default="mspc",
    show_default=True,
    help="STAC backend to query.",
)
@click.option(
    "--s2-collection",
    default="c1",
    show_default=True,
    help="Sentinel-2 collection (EarthSearch only).",
)
@click.option(
    "--cloud-cover-max",
    type=click.IntRange(min=0, max=100),
    default=10,
    show_default=True,
    help="Maximum cloud cover percentage per scene.",
)
@click.option(
    "--buffer-days",
    type=click.IntRange(min=0),
    default=14,
    show_default=True,
    help="Search window around crop-calendar dates (days).",
)
@click.option(
    "--nodata-max",
    type=click.IntRange(min=0, max=100),
    default=30,
    show_default=True,
    help="Maximum nodata percentage per scene.",
)
@click.option(
    "--timeout",
    type=click.IntRange(min=1),
    default=120,
    show_default=True,
    help="Seconds before the create-input download is aborted.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Enable verbose output."
)
def main(
    bbox,
    geojson_path,
    year,
    output,
    do_create_input,
    stac_host,
    s2_collection,
    cloud_cover_max,
    buffer_days,
    nodata_max,
    timeout,
    verbose,
):
    """Fetch Sentinel-2 planting/harvest scene pairs (win_a, win_b) for an AOI.

    Pass --create-input to additionally download and merge the two scenes
    into a single 8-band GeoTIFF suitable for FTW inference.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    if bbox is None and geojson_path is None:
        raise click.UsageError("Provide either --bbox or --geojson.")

    # ------------------------------------------------------------------
    # Resolve final bounding box
    # ------------------------------------------------------------------
    if geojson_path is not None:
        aoi = gpd.read_file(geojson_path)
        if aoi.crs is None:
            aoi = aoi.set_crs("EPSG:4326")
        else:
            aoi = aoi.to_crs("EPSG:4326")
        gj_xmin, gj_ymin, gj_xmax, gj_ymax = aoi.total_bounds
        if bbox is not None:
            xmin = max(bbox[0], gj_xmin)
            ymin = max(bbox[1], gj_ymin)
            xmax = min(bbox[2], gj_xmax)
            ymax = min(bbox[3], gj_ymax)
        else:
            xmin, ymin, xmax, ymax = gj_xmin, gj_ymin, gj_xmax, gj_ymax
        click.echo(f"Derived bbox from GeoJSON: {xmin},{ymin},{xmax},{ymax}")
    else:
        xmin, ymin, xmax, ymax = bbox

    resolved_bbox = [xmin, ymin, xmax, ymax]

    # ------------------------------------------------------------------
    # Scene selection
    # ------------------------------------------------------------------
    click.echo(f"Querying STAC ({stac_host}) for year={year}, bbox={resolved_bbox} ...")
    dask.config.set({"array.chunk-size": "256MiB"})

    try:
        win_a, win_b = scene_selection(
            bbox=resolved_bbox,
            year=year,
            stac_host=stac_host,
            cloud_cover_max=cloud_cover_max,
            buffer_days=buffer_days,
            s2_collection=s2_collection,
            nodata_max=nodata_max,
            verbose=verbose,
        )
    except Exception as exc:
        raise click.ClickException(f"Scene selection failed: {exc}") from exc

    click.echo(f"  win_a (early season / planting): {win_a}")
    click.echo(f"  win_b (late season  / harvest) : {win_b}")

    if not do_create_input:
        return

    # ------------------------------------------------------------------
    # Download & merge
    # ------------------------------------------------------------------
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Downloading and merging scenes to {output} (timeout={timeout}s) ...")

    # Register SIGALRM-based timeout (Linux/macOS only)
    if sys.platform != "win32":
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)

    try:
        create_input(
            win_a=win_a,
            win_b=win_b,
            out=str(output),
            overwrite=True,
            stac_host=stac_host,
            bbox=resolved_bbox,
            s2_collection=s2_collection,
            verbose=verbose,
        )
        if sys.platform != "win32":
            signal.alarm(0)  # cancel alarm on success
    except _DownloadTimeout:
        raise click.ClickException(
            f"Download exceeded {timeout}s timeout — output not written."
        )
    except Exception as exc:
        if sys.platform != "win32":
            signal.alarm(0)
        raise click.ClickException(f"create_input failed: {exc}") from exc

    click.echo(f"Done. Output: {output}")


if __name__ == "__main__":
    main()

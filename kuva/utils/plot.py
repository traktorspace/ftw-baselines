import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio

CLASS_COLORS = {
    0: ("#2d6a4f", "background"),
    1: ("#f4a261", "field boundary"),
    2: ("#e76f51", "field interior"),
}

FIGSIZE_PER_COL = 5
NCOLS = 2


def _make_legend_patches(unique_vals):
    patches = []
    for v in sorted(unique_vals):
        if v in CLASS_COLORS:
            color, label = CLASS_COLORS[v]
            patches.append(mpatches.Patch(color=color, label=f"{label} ({v})"))
        else:
            patches.append(mpatches.Patch(label=f"class {v}"))
    return patches


def _apply_class_colormap(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*arr.shape, 3), dtype=np.float32)
    for val, (hex_color, _) in CLASS_COLORS.items():
        r, g, b = tuple(
            int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)
        )
        rgb[arr == val] = [r, g, b]
    return rgb


def create_plot_inference_results(
    prediction_tifs: list[str | Path],
    source_tif: str | Path | None = None,
    title: str = "",
    figsize_per_col: int = FIGSIZE_PER_COL,
    ncols: int = NCOLS,
) -> plt.Figure:
    """Display inference results, optionally alongside the source image.

    The source TIF, when provided, must have either 6 or 8 bands:
      - 6 bands: RGB t0 (bands 1-3) + RGB t1 (bands 4-6)
      - 8 bands: RGBN t0 (bands 1-4) + RGBN t1 (bands 5-8); only the RGB channels are displayed

    Args:
        prediction_tifs: List of paths to prediction .tif files to display.
        source_tif: Optional path to the source .tif file with 6 or 8 bands (t0/t1 pair).
            If not provided, only the predictions are plotted.
        title: Title for the overall figure. Defaults to the source file stem if provided.
        figsize_per_col: Width and height in inches per subplot column.
        ncols: Number of columns in the prediction grid.

    Returns:
        The matplotlib Figure.

    Raises:
        AssertionError: If the source TIF does not have 6 or 8 bands.
    """
    prediction_tifs = [Path(p) for p in prediction_tifs]

    n_results = len(prediction_tifs)
    pred_rows = math.ceil(n_results / ncols) if n_results > 0 else 0
    source_row = 1 if source_tif is not None else 0
    nrows = source_row + pred_rows

    fig, ax = plt.subplots(
        nrows, ncols, figsize=(figsize_per_col * ncols, figsize_per_col * nrows)
    )
    ax = np.atleast_2d(ax)

    if title:
        fig_title = title
    elif source_tif is not None:
        fig_title = Path(source_tif).stem
    else:
        fig_title = ""
    if fig_title:
        fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.01)

    # --- Optional row 0: t0 and t1 RGB ---
    if source_tif is not None:
        source_tif = Path(source_tif)
        with rasterio.open(source_tif) as src:
            data = src.read()

        n_bands = data.shape[0]
        assert n_bands in (6, 8), f"Expected 6 or 8 bands, got {n_bands}."
        t1_start = 4 if n_bands == 8 else 3

        t0_img = np.clip(data[0:3].transpose(1, 2, 0) / 3000, 0, 1)
        ax[0, 0].imshow(t0_img)
        ax[0, 0].set_title("t0", fontsize=10)
        ax[0, 0].axis("off")

        t1_img = np.clip(data[t1_start : t1_start + 3].transpose(1, 2, 0) / 3000, 0, 1)
        ax[0, 1].imshow(t1_img)
        ax[0, 1].set_title("t1", fontsize=10)
        ax[0, 1].axis("off")

    # --- Prediction rows ---
    for i, pred_path in enumerate(prediction_tifs):
        row = source_row + i // ncols
        col = i % ncols

        with rasterio.open(pred_path) as inf_src:
            out_dat = inf_src.read()

        if out_dat.shape[0] == 1:
            arr = out_dat.squeeze(0)
            rgb = _apply_class_colormap(arr)
            ax[row, col].imshow(rgb, interpolation="nearest")
            patches = _make_legend_patches(np.unique(arr))
            ax[row, col].legend(
                handles=patches,
                loc="lower right",
                fontsize=7,
                framealpha=0.8,
                edgecolor="gray",
            )
        else:
            img = np.moveaxis(out_dat, 0, -1)
            ax[row, col].imshow(np.clip(img, 0, 1))

        ax[row, col].set_title(pred_path.stem, fontsize=9)
        ax[row, col].axis("off")

    # Hide unused cells in the last prediction row
    for j in range(n_results % ncols or ncols, ncols):
        ax[-1, j].axis("off")

    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_bitemporal(tif_path: str, norm: float = 3000.0) -> plt.Figure:
    """Plot a bitemporal RGB acquisition from a multi-band TIF.

    Expects either 6-band (RGB t0 + RGB t1) or 8-band (RGBN t0 + RGBN t1) input.

    Args:
        tif_path: Path to the source TIF file.
        norm: Normalization factor for reflectance values.

    Returns:
        The matplotlib Figure.
    """
    path = Path(tif_path)
    with rasterio.open(tif_path) as src:
        n_bands = src.count
        assert n_bands in (6, 8), f"Expected 6 or 8 bands, got {n_bands}"
        stride = n_bands // 2  # 3 or 4
        t0 = src.read((1, 2, 3)).transpose(1, 2, 0) / norm
        t1 = src.read((stride + 1, stride + 2, stride + 3)).transpose(1, 2, 0) / norm

    t0 = np.clip(t0, 0, 1)
    t1 = np.clip(t1, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(path.name.stem, fontsize=13, fontweight="bold")

    for ax, img, title in zip(axes, [t0, t1], ["t0", "t1"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.close(fig)
    return fig

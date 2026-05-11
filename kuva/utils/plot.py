import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio

CLASS_COLORS = {
    0: ("#2d6a4f", "background"),
    1: ("#f4a261", "field interior"),
    2: ("#e76f51", "field boundary"),
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
    crop: tuple[int, int, int, int] | None = None,
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
        crop: Optional spatial subset as ``(row_start, row_end, col_start, col_end)``.
            Applied to both the source and prediction TIFs. Defaults to the full image.

    Returns:
        The matplotlib Figure.

    Raises:
        AssertionError: If the source TIF does not have 6 or 8 bands.
    """
    prediction_tifs = [Path(p) for p in prediction_tifs]

    window: rasterio.windows.Window | None = None
    if crop is not None:
        row_start, row_end, col_start, col_end = crop
        window = rasterio.windows.Window(
            col_off=col_start,
            row_off=row_start,
            width=col_end - col_start,
            height=row_end - row_start,
        )

    n_results = len(prediction_tifs)
    pred_rows = math.ceil(n_results / ncols) if n_results > 0 else 0
    source_row = 1 if source_tif is not None else 0
    nrows = source_row + pred_rows

    fig = plt.figure(figsize=(figsize_per_col * ncols, figsize_per_col * nrows))
    gs_outer = fig.add_gridspec(nrows, 1, hspace=0.3)

    if title:
        fig_title = title
    elif source_tif is not None:
        fig_title = Path(source_tif).stem
    else:
        fig_title = ""
    if fig_title:
        fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=1.01)

    # --- Optional row 0: t0 and t1 RGB — always 2 equal columns ---
    if source_tif is not None:
        source_tif = Path(source_tif)
        with rasterio.open(source_tif) as src:
            data = src.read(window=window)

        n_bands = data.shape[0]
        assert n_bands in (6, 8), f"Expected 6 or 8 bands, got {n_bands}."
        t1_start = 4 if n_bands == 8 else 3

        src_gs = gs_outer[0].subgridspec(1, 2, wspace=0.05)
        ax_t0 = fig.add_subplot(src_gs[0, 0])
        ax_t1 = fig.add_subplot(src_gs[0, 1])

        t0_img = np.clip(data[0:3].transpose(1, 2, 0) / 3000, 0, 1)
        ax_t0.imshow(t0_img)
        ax_t0.set_title("t0", fontsize=10)
        ax_t0.axis("off")

        t1_img = np.clip(data[t1_start : t1_start + 3].transpose(1, 2, 0) / 3000, 0, 1)
        ax_t1.imshow(t1_img)
        ax_t1.set_title("t1", fontsize=10)
        ax_t1.axis("off")

    # --- Prediction rows — ncols columns each ---
    pred_axes: dict[tuple[int, int], plt.Axes] = {}
    for r in range(pred_rows):
        pred_gs = gs_outer[source_row + r].subgridspec(1, ncols, wspace=0.05)
        for c in range(ncols):
            pred_axes[(r, c)] = fig.add_subplot(pred_gs[0, c])

    for i, pred_path in enumerate(prediction_tifs):
        r, c = i // ncols, i % ncols
        ax = pred_axes[(r, c)]

        with rasterio.open(pred_path) as inf_src:
            out_dat = inf_src.read(window=window)

        if out_dat.shape[0] == 1:
            arr = out_dat.squeeze(0)
            rgb = _apply_class_colormap(arr)
            ax.imshow(rgb, interpolation="nearest")
            patches = _make_legend_patches(np.unique(arr))
            ax.legend(
                handles=patches,
                loc="lower right",
                fontsize=7,
                framealpha=0.8,
                edgecolor="gray",
            )
        else:
            img = np.moveaxis(out_dat, 0, -1)
            ax.imshow(np.clip(img, 0, 1))

        ax.set_title(pred_path.stem, fontsize=9)
        ax.axis("off")

    # Hide unused cells in the last prediction row
    last_row = pred_rows - 1
    for c in range(n_results % ncols or ncols, ncols):
        pred_axes[(last_row, c)].axis("off")

    plt.close(fig)
    return fig


def plot_class_distribution(
    prediction_tifs: list[str | Path],
    crop: tuple[int, int, int, int] | None = None,
    figsize: tuple[float, float] | None = None,
    ncols: int = NCOLS,
    show_sorted: bool = False,
) -> "tuple[plt.Figure, dict] | tuple[plt.Figure, plt.Figure, dict]":
    """Plot class pixel fractions for a list of prediction TIFs.

    Always renders a per-file grid (one subplot per file, bars per class).
    When ``show_sorted=True``, also renders a second figure with one subplot per
    class, files sorted from highest to lowest fraction.

    Args:
        prediction_tifs: List of paths to single-band label prediction .tif files.
        crop: Optional spatial subset as ``(row_start, row_end, col_start, col_end)``.
        figsize: Figure size ``(width, height)`` in inches for the per-file grid.
            Auto-sized if *None*.
        ncols: Number of subplot columns in the per-file grid.
        show_sorted: If *True*, also build the sorted figure.

    Returns:
        ``(fig_grid, fractions)`` when ``show_sorted=False``, or
        ``(fig_grid, fig_sorted, fractions)`` when ``show_sorted=True``.
        ``fractions`` is a ``dict[str, dict[int, float]]`` mapping each file stem
        to a ``{class_id: fraction}`` dict.
    """
    prediction_tifs = [Path(p) for p in prediction_tifs]
    n = len(prediction_tifs)

    window: rasterio.windows.Window | None = None
    if crop is not None:
        row_start, row_end, col_start, col_end = crop
        window = rasterio.windows.Window(
            col_off=col_start,
            row_off=row_start,
            width=col_end - col_start,
            height=row_end - row_start,
        )

    classes = sorted(CLASS_COLORS.keys())
    n_classes = len(classes)

    # --- collect fractions for every file once ---
    all_fractions: dict[int, list[tuple[str, float]]] = {c: [] for c in classes}
    for pred_path in prediction_tifs:
        with rasterio.open(pred_path) as src:
            arr = src.read(1, window=window)
        total = arr.size
        for c in classes:
            all_fractions[c].append((pred_path.stem, np.sum(arr == c) / total))

    bar_colors = [CLASS_COLORS[c][0] for c in classes]
    bar_labels = [CLASS_COLORS[c][1] for c in classes]

    # --- per-file grid figure ---
    nrows = math.ceil(n / ncols)
    grid_figsize = figsize
    if grid_figsize is None:
        w = 4.0 * min(n, ncols)
        h = 4 * nrows
        grid_figsize = (max(w, 8), max(h, 4))

    fig_grid, axes = plt.subplots(
        nrows, ncols, figsize=grid_figsize, squeeze=False, constrained_layout=True
    )

    for i, pred_path in enumerate(prediction_tifs):
        ax = axes[i // ncols][i % ncols]
        fractions = [all_fractions[cls][i][1] for cls in classes]

        bars = ax.bar(
            bar_labels, fractions, color=bar_colors, edgecolor="white", width=0.5
        )
        for bar, frac in zip(bars, fractions):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                frac + 0.01,
                f"{frac:.1%}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction", fontsize=8)
        ax.set_title(pred_path.stem, fontsize=8)
        ax.tick_params(axis="x", labelsize=8, rotation=15)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    plt.close(fig_grid)

    # Build fractions dict: {stem: {class_id: fraction}}
    fractions_out: dict[str, dict[int, float]] = {
        pred_path.stem: {cls: all_fractions[cls][i][1] for cls in classes}
        for i, pred_path in enumerate(prediction_tifs)
    }

    if not show_sorted:
        return fig_grid, fractions_out

    # --- sorted figure ---
    max_stem_len = max((len(p.stem) for p in prediction_tifs), default=10)
    label_inches = max_stem_len * 0.07
    row_h = 4 + label_inches
    sorted_figsize = (max(n * 0.8, 8), row_h * n_classes)

    fig_sorted, sorted_axes = plt.subplots(n_classes, 1, figsize=sorted_figsize)
    if n_classes == 1:
        sorted_axes = [sorted_axes]

    for ax, cls in zip(sorted_axes, classes):
        color, label = CLASS_COLORS[cls]
        sorted_pairs = sorted(all_fractions[cls], key=lambda x: x[1], reverse=True)
        stems, vals = zip(*sorted_pairs)
        display_stems = [st[-40:] if len(st) > 40 else st for st in stems]

        bars = ax.bar(display_stems, vals, color=color, edgecolor="white", width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.005,
                f"{val:.1%}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction", fontsize=8)
        ax.set_title(f"{label} (class {cls}) — sorted", fontsize=10)
        ax.set_xticks(range(len(display_stems)))
        ax.set_xticklabels(display_stems, ha="right", rotation=40, fontsize=7)

    fig_sorted.tight_layout(h_pad=3.0)
    plt.close(fig_sorted)

    return fig_grid, fig_sorted, fractions_out


def plot_bitemporal(
    tif_path: str, norm: float = 3000.0, crop: tuple[int, int, int, int] | None = None
) -> plt.Figure:
    """Plot a bitemporal RGB acquisition from a multi-band TIF.

    Expects either 6-band (RGB t0 + RGB t1) or 8-band (RGBN t0 + RGBN t1) input.

    Args:
        tif_path: Path to the source TIF file.
        norm: Normalization factor for reflectance values.
        crop: Optional spatial subset as ``(row_start, row_end, col_start, col_end)``.
            Example: ``(1000, 2000, 1000, 2000)``.  Defaults to the full image.

    Returns:
        The matplotlib Figure.
    """
    path = Path(tif_path)
    with rasterio.open(tif_path) as src:
        n_bands = src.count
        assert n_bands in (6, 8), f"Expected 6 or 8 bands, got {n_bands}"
        stride = n_bands // 2  # 3 or 4
        if crop is not None:
            row_start, row_end, col_start, col_end = crop
            window = rasterio.windows.Window(
                col_off=col_start,
                row_off=row_start,
                width=col_end - col_start,
                height=row_end - row_start,
            )
            t0 = src.read((1, 2, 3), window=window).transpose(1, 2, 0) / norm
            t1 = (
                src.read((stride + 1, stride + 2, stride + 3), window=window).transpose(
                    1, 2, 0
                )
                / norm
            )
        else:
            t0 = src.read((1, 2, 3)).transpose(1, 2, 0) / norm
            t1 = (
                src.read((stride + 1, stride + 2, stride + 3)).transpose(1, 2, 0) / norm
            )

    t0 = np.clip(t0, 0, 1)
    t1 = np.clip(t1, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(path.stem, fontsize=13, fontweight="bold")

    for ax, img, title in zip(axes, [t0, t1], ["t0", "t1"]):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.close(fig)
    return fig

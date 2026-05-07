"""
Classical watershed segmentation baseline implemented as a ``torch.nn.Module``.

The pipeline is:

1. Per-channel min-max normalisation (with an optional hard clip).
2. Gradient computation via Scharr or Canny using *kornia* (GPU-compatible).
3. Watershed hierarchy cut at a minimum-area threshold using *higra* (CPU/NumPy).

Dependencies
------------
torch, kornia, higra, rasterio
"""

import higra as hg
import kornia.filters as KF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WatershedBaselineSegmentation(nn.Module):
    """Watershed-based field boundary segmentation baseline.

    Accepts a single multi-spectral image tile, computes an edge gradient, and
    returns both the gradient map and a labelled segment map produced by a
    watershed hierarchy cut.

    Parameters
    ----------
    mode : {"per_band", "ndvi", "per_band+ndvi"}, optional
        Gradient source strategy.

        ``"per_band"``
            Apply the filter to every spectral band independently and take the
            pixel-wise maximum across bands.
        ``"ndvi"``
            Compute NDVI from ``nir_band`` and ``red_band``, then apply the
            filter to the resulting single-channel index image.
        ``"per_band+ndvi"``
            Combine both strategies by taking the pixel-wise maximum of the
            per-band and NDVI gradients.
    filter_type : {"scharr", "canny"}, optional
        Edge-detection operator.

        ``"scharr"``
            Scharr gradient magnitude — continuous values in ``[0, 1]``,
            better suited for watershed because it provides smooth boundaries.
        ``"canny"``
            Canny binary edge map — values in ``{0, 1}``.
    canny_low : float, optional
        Lower hysteresis threshold for the Canny detector, in ``[0, 1]``
        (kornia convention).  Only used when ``filter_type="canny"``.
    canny_high : float, optional
        Upper hysteresis threshold for the Canny detector, in ``[0, 1]``.
        Must satisfy ``canny_low < canny_high``.
    min_area : int, optional
        Minimum number of pixels for a watershed segment.  Segments smaller
        than this threshold are merged into their lower-altitude neighbour
        during the horizontal cut.
    nir_band : int, optional
        Zero-based index of the NIR band in the input tensor.  Used only when
        ``mode`` includes ``"ndvi"``.
    red_band : int, optional
        Zero-based index of the Red band in the input tensor.  Used only when
        ``mode`` includes ``"ndvi"``.
    """

    def __init__(
        self,
        mode: str = "per_band",
        filter_type: str = "scharr",
        canny_low: float = 0.1,
        canny_high: float = 0.2,
        min_area: int = 200,
        nir_band: int = 3,
        red_band: int = 2,
        tile_size: int | None = None,
        tile_overlap: int = 64,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        super().__init__()

        if mode not in ("per_band", "ndvi", "per_band+ndvi"):
            raise ValueError(
                f"mode must be 'per_band', 'ndvi', or 'per_band+ndvi', got '{mode}'"
            )
        if mode in ("ndvi", "per_band+ndvi"):
            if nir_band == red_band:
                raise ValueError("nir_band and red_band must differ")
        if filter_type not in ("scharr", "canny"):
            raise ValueError(
                f"filter_type must be 'scharr' or 'canny', got {filter_type}"
            )
        if not (0.0 <= canny_low < canny_high <= 1.0):
            raise ValueError("need 0 <= canny_low < canny_high <= 1")
        if tile_size is not None and tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        self.mode = mode
        self.filter_type = filter_type
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_area
        self.nir_band = nir_band
        self.red_band = red_band
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.n_jobs = n_jobs
        self.verbose = verbose
        # Register Scharr kernels as buffers so they live on the correct device
        # and are not re-allocated on every forward pass.
        scharr_x = (
            torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]).view(
                1, 1, 3, 3
            )
            / 16.0
        )
        self.register_buffer("_scharr_x", scharr_x)
        self.register_buffer("_scharr_y", scharr_x.transpose(-1, -2).contiguous())

    # ── internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _channel_normalize(x: torch.Tensor, clip: float = 2000.0) -> torch.Tensor:
        """Normalise each spectral band independently to ``[0, 1]``.

        Pixel values are first clipped to ``[0, clip]`` and then rescaled via
        per-channel min-max normalisation.  This mirrors the
        ``channel_normalize()`` function used in FAUNet preprocessing so that
        both pipelines operate on the same value range.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(1, C, H, W)``, float32.
        clip : float, optional
            Hard upper clip applied before normalisation.  Values above this
            threshold (e.g. cloud-contaminated pixels) are clamped to ``clip``.

        Returns
        -------
        torch.Tensor
            Normalised tensor of shape ``(1, C, H, W)``, float32 in ``[0, 1]``.
        """
        x = x.nan_to_num(nan=0.0, posinf=clip, neginf=0.0)
        x = x.clamp(max=clip)
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)  # (1, C, H*W)
        x_min = x_flat.min(dim=-1, keepdim=True).values  # (1, C, 1)
        x_max = x_flat.max(dim=-1, keepdim=True).values
        denom = (x_max - x_min).clamp(min=1e-6)
        x_norm = (x_flat - x_min) / denom
        return x_norm.view(b, c, h, w)

    def _scharr(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Scharr gradient magnitude for a single-channel image.

        The Scharr operator is a refinement of Sobel that achieves better
        rotational symmetry.  The normalised horizontal kernel is:

        .. code-block:: none

            K_x = (1/16) * [[ -3,  0,  3],
                            [-10,  0, 10],
                            [ -3,  0,  3]]

        and ``K_y = K_x^T``.  The positive weights sum to 16, so dividing by
        16 keeps responses in ``[-1, 1]`` for input images in ``[0, 1]``.

        The full computation is:

        1. Gaussian pre-blur (5x5, σ=1.0) to suppress high-frequency noise.
        2. Directional responses: ``G_x = I * K_x``, ``G_y = I * K_y``.
        3. Gradient magnitude: ``|G| = sqrt(G_x² + G_y²)``.
        4. Spatial normalisation: ``output = |G| / max(|G|)``.

        Parameters
        ----------
        x : torch.Tensor
            Single-channel image of shape ``(1, 1, H, W)``, float32 in
            ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Gradient magnitude of shape ``(1, 1, H, W)``, float32 in
            ``[0, 1]``.
        """
        x = KF.gaussian_blur2d(x, kernel_size=(5, 5), sigma=(1.0, 1.0))
        gx = F.conv2d(x, self._scharr_x, padding=1)
        gy = F.conv2d(x, self._scharr_y, padding=1)
        mag = (gx.pow(2) + gy.pow(2)).sqrt()
        mag_max = mag.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        return mag / mag_max

    def _canny(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a binary Canny edge map for a single-channel image.

        Delegates to ``kornia.filters.canny``, which returns a tuple
        ``(edges, magnitude)``.  Only the binary edge map is used.
        Thresholds are taken from ``self.canny_low`` and ``self.canny_high``.

        Parameters
        ----------
        x : torch.Tensor
            Single-channel image of shape ``(1, 1, H, W)``, float32.

        Returns
        -------
        torch.Tensor
            Binary edge map of shape ``(1, 1, H, W)``, float32 in ``{0, 1}``.
        """
        edges, _ = KF.canny(
            x, low_threshold=self.canny_low, high_threshold=self.canny_high
        )
        return edges.float()

    def _apply_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to the selected edge-detection filter.

        Parameters
        ----------
        x : torch.Tensor
            Single-channel image of shape ``(1, 1, H, W)``, float32.

        Returns
        -------
        torch.Tensor
            Edge/gradient response of shape ``(1, 1, H, W)``, float32.
        """
        if self.filter_type == "scharr":
            return self._scharr(x)
        else:
            return self._canny(x)

    def _gradient_per_band(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the gradient for each spectral band and take the pixel-wise maximum.

        Each band is processed independently through ``_apply_filter``.  The
        final gradient highlights any band that shows a strong edge at a given
        pixel.

        Parameters
        ----------
        x : torch.Tensor
            Multi-band image of shape ``(1, C, H, W)``, float32 in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Combined gradient of shape ``(1, 1, H, W)``, float32.
        """
        b, c, h, w = x.shape
        if self.filter_type == "scharr":
            # Single batched GPU pass over all bands: (b*c, 1, H, W)
            x_bc = x.view(b * c, 1, h, w)
            x_bc = KF.gaussian_blur2d(x_bc, kernel_size=(5, 5), sigma=(1.0, 1.0))
            gx = F.conv2d(x_bc, self._scharr_x, padding=1)
            gy = F.conv2d(x_bc, self._scharr_y, padding=1)
            mag = (gx.pow(2) + gy.pow(2)).sqrt().view(b, c, h, w)
            mag_max = mag.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
            return (mag / mag_max).max(dim=1, keepdim=True).values
        else:  # canny – kornia also supports batched input
            x_bc = x.view(b * c, 1, h, w)
            edges, _ = KF.canny(
                x_bc, low_threshold=self.canny_low, high_threshold=self.canny_high
            )
            return edges.float().view(b, c, h, w).max(dim=1, keepdim=True).values

    def _gradient_ndvi(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the gradient of the NDVI index.

        NDVI is computed as ``(NIR - Red) / (NIR + Red)``, mapped from
        ``[-1, 1]`` to ``[0, 1]``, and then passed through ``_apply_filter``.
        Band indices are taken from ``self.nir_band`` and ``self.red_band``.

        Parameters
        ----------
        x : torch.Tensor
            Multi-band image of shape ``(1, C, H, W)``, float32 in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            NDVI gradient of shape ``(1, 1, H, W)``, float32.
        """
        nir = x[:, self.nir_band : self.nir_band + 1]  # (1, 1, H, W)
        red = x[:, self.red_band : self.red_band + 1]
        denom = (nir + red).clamp(min=1e-6)
        ndvi = (nir - red) / denom  # [-1, 1]
        ndvi_norm = (ndvi + 1.0) / 2.0  # → [0, 1]
        return self._apply_filter(ndvi_norm)

    def _gradient_per_band_ndvi(self, x: torch.Tensor) -> torch.Tensor:
        """Combine per-band and NDVI gradients by pixel-wise maximum.

        Merges the outputs of :meth:`_gradient_per_band` and
        :meth:`_gradient_ndvi`, capturing both generic spectral boundaries and
        vegetation-specific edges in a single gradient map.

        Parameters
        ----------
        x : torch.Tensor
            Multi-band image of shape ``(1, C, H, W)``, float32 in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Combined gradient of shape ``(1, 1, H, W)``, float32.
        """
        grad_per_band = self._gradient_per_band(x)  # (1, 1, H, W)
        grad_ndvi = self._gradient_ndvi(x)  # (1, 1, H, W)
        return torch.max(grad_per_band, grad_ndvi)

    @staticmethod
    def _watershed(gradient_np: np.ndarray, min_area: int) -> np.ndarray:
        """Run a watershed hierarchy and return a labelled segment map.

        Uses *higra*'s area-based watershed hierarchy on a 4-adjacency graph.
        Edge weights are the mean gradient of the two adjacent pixels (mean
        rather than max to reduce oversegmentation).  The hierarchy is then
        cut at ``min_area`` to merge small segments.  Labels are shifted by
        +1 so that 0 is reserved for background / no-data.

        Parameters
        ----------
        gradient_np : np.ndarray
            Gradient image of shape ``(H, W)``, float32.
        min_area : int
            Minimum segment area in pixels.  Segments smaller than this are
            merged during the horizontal cut.

        Returns
        -------
        np.ndarray
            Integer label map of shape ``(H, W)``, int32.  Valid segments are
            labelled starting from 1; 0 is reserved for no-data.
        """
        h, w = gradient_np.shape
        graph = hg.get_4_adjacency_graph((h, w))
        edge_weights = hg.weight_graph(
            graph, gradient_np, hg.WeightFunction.mean
        )  # mean instead of max to reduce oversegmentation
        tree, altitudes = hg.watershed_hierarchy_by_area(graph, edge_weights)
        labels = hg.labelisation_horizontal_cut_from_threshold(
            tree, altitudes, min_area
        )
        # After labelisation, shift labels so 0 is free for background
        labels = labels + 1  # valid segments start at 1
        labels = labels.reshape(h, w).astype(np.int32)
        return labels

    @staticmethod
    def _watershed_tiled(
        gradient_np: np.ndarray,
        min_area: int,
        tile_size: int,
        overlap: int,
        n_jobs: int,
        verbose: bool = False,
    ) -> np.ndarray:
        """Tiled watershed for large images with optional parallel execution.

        Splits the gradient into overlapping tiles, runs :meth:`_watershed` on
        each tile (optionally in parallel via threads, which works because
        *higra* releases the GIL), then stitches the per-tile label maps.
        Labels are offset per tile so they never collide globally.

        .. note::
            Segments that straddle a tile boundary will be split into two
            separate labels.  Choose a ``tile_size`` larger than the biggest
            expected field and an ``overlap`` of at least half that field size
            to minimise boundary artefacts.

        Parameters
        ----------
        gradient_np : np.ndarray
            Gradient image of shape ``(H, W)``, float32.
        min_area : int
            Minimum segment area forwarded to :meth:`_watershed`.
        tile_size : int
            Height and width of each interior tile in pixels.
        overlap : int
            Extra pixels added on each side before running the watershed.
            Only the interior portion is kept in the output.
        n_jobs : int
            Maximum number of worker threads.  Use ``1`` to disable parallelism.

        Returns
        -------
        np.ndarray
            Integer label map of shape ``(H, W)``, int32.
        """
        from concurrent.futures import ThreadPoolExecutor

        try:
            from tqdm.auto import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        h, w = gradient_np.shape

        tiles: list[tuple[int, int, int, int, int, int, int, int]] = []
        for rs in range(0, h, tile_size):
            re = min(rs + tile_size, h)
            ers = max(0, rs - overlap)
            ere = min(h, re + overlap)
            for cs in range(0, w, tile_size):
                ce = min(cs + tile_size, w)
                ecs = max(0, cs - overlap)
                ece = min(w, ce + overlap)
                tiles.append((rs, re, cs, ce, ers, ere, ecs, ece))

        def _process(t: tuple) -> np.ndarray:
            rs, re, cs, ce, ers, ere, ecs, ece = t
            tile_labels = WatershedBaselineSegmentation._watershed(
                gradient_np[ers:ere, ecs:ece], min_area
            )
            dr, dc = rs - ers, cs - ecs
            return tile_labels[dr : dr + (re - rs), dc : dc + (ce - cs)]

        n_tiles = len(tiles)
        desc = f"Watershed tiles ({n_tiles})"
        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                it = pool.map(_process, tiles)
                if verbose and _tqdm is not None:
                    it = _tqdm(it, total=n_tiles, desc=desc)
                tile_results = list(it)
        else:
            it = tiles
            if verbose and _tqdm is not None:
                it = _tqdm(it, total=n_tiles, desc=desc)
            tile_results = [_process(t) for t in it]

        output = np.zeros((h, w), dtype=np.int32)
        label_offset = 0
        seam_rows: list[int] = []
        seam_cols: list[int] = []
        for (rs, re, cs, ce, *_), interior in zip(tiles, tile_results):
            mask = interior > 0
            if mask.any():
                shifted = interior.copy()
                shifted[mask] += label_offset
                output[rs:re, cs:ce] = shifted
                label_offset += int(interior.max())
            else:
                output[rs:re, cs:ce] = interior
            if re < h and re not in seam_rows:
                seam_rows.append(re)
            if ce < w and ce not in seam_cols:
                seam_cols.append(ce)

        return WatershedBaselineSegmentation._merge_seam_labels(
            output, seam_rows, seam_cols
        )

    @staticmethod
    def _merge_seam_labels(
        labels: np.ndarray, seam_rows: list[int], seam_cols: list[int]
    ) -> np.ndarray:
        """Merge labels that are adjacent across tile seams using union-find.

        For every seam (row or column boundary between tiles) any pair of
        adjacent non-zero labels on opposite sides of the seam is unified.
        The result is relabelled compactly so downstream code sees a clean,
        contiguous label space.

        Parameters
        ----------
        labels : np.ndarray
            Stitched label map of shape ``(H, W)``, int32.
        seam_rows : list[int]
            Row indices at which horizontal tile seams occur.
        seam_cols : list[int]
            Column indices at which vertical tile seams occur.

        Returns
        -------
        np.ndarray
            Relabelled map of shape ``(H, W)``, int32.
        """
        max_label = int(labels.max()) + 1
        parent = np.arange(max_label, dtype=np.int32)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Merge across horizontal seams (look one row above and below the seam)
        for r in seam_rows:
            la = labels[r - 1, :]
            lb = labels[r, :]
            mask = (la > 0) & (lb > 0)
            for a, b in zip(la[mask].tolist(), lb[mask].tolist()):
                union(a, b)

        # Merge across vertical seams
        for c in seam_cols:
            la = labels[:, c - 1]
            lb = labels[:, c]
            mask = (la > 0) & (lb > 0)
            for a, b in zip(la[mask].tolist(), lb[mask].tolist()):
                union(a, b)

        # Build compact remap: label → new sequential id
        roots = np.array([find(i) for i in range(max_label)], dtype=np.int32)
        unique_roots, inverse = np.unique(roots, return_inverse=True)
        # Preserve 0 as background (find(0)==0 always since 0 is never merged)
        remap = inverse.astype(np.int32)

        flat = labels.ravel()
        return remap[roots[flat]].reshape(labels.shape)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """Run the full segmentation pipeline on a single image tile.

        Steps:

        1. Per-channel min-max normalisation (clip at 2000).
        2. Gradient computation according to ``self.mode``.
        3. No-data pixels (all bands ≤ 0) are forced to gradient = 1 so that
           the watershed treats them as hard boundaries.
        4. Watershed hierarchy cut at ``self.min_area``.
        5. No-data pixels are set to label 0 in the output.

        Parameters
        ----------
        x : torch.Tensor
            Raw multi-spectral image of shape ``(1, C, H, W)``, float32.

        Returns
        -------
        gradient : torch.Tensor
            Edge gradient of shape ``(1, 1, H, W)``, float32 in ``[0, 1]``.
        labels : np.ndarray
            Segment label map of shape ``(H, W)``, int32.  Valid segments
            start at 1; 0 indicates no-data / background.
        """
        assert x.dim() == 4 and x.shape[0] == 1, "expected (1, C, H, W)"

        # 1. Normalise (same logic as FAUNet preprocessing)
        x_norm = self._channel_normalize(x)

        # 2. Compute gradient
        if self.mode == "per_band":
            gradient = self._gradient_per_band(x_norm)
        elif self.mode == "ndvi":
            gradient = self._gradient_ndvi(x_norm)
        else:  # "per_band+ndvi"
            gradient = self._gradient_per_band_ndvi(x_norm)

        # 3. Watershed (higra works on numpy)
        grad_np = gradient.squeeze().detach().cpu().numpy().astype(np.float32)
        # 4. No-data mask: pixels where all bands are zero/NaN (or near-zero)
        nodata_mask = (
            ((x <= 0) | torch.isnan(x)).all(dim=1).squeeze(0).cpu().numpy()
        )  # (H, W) bool

        # Force maximum gradient at no-data pixels → watershed keeps them as boundaries
        grad_np[nodata_mask] = 1.0

        # 5. Watershed
        if self.tile_size is not None:
            labels = self._watershed_tiled(
                grad_np,
                self.min_area,
                self.tile_size,
                self.tile_overlap,
                self.n_jobs,
                self.verbose,
            )
        else:
            labels = self._watershed(grad_np, self.min_area)

        # 6. Mask out no-data from labels (reserve label 0 for background/no-data)
        labels[nodata_mask] = 0

        return gradient, labels

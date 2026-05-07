"""FAUNet RGB inference script.

Runs inference with a FAUNet checkpoint on a single GeoTIFF input.
FAUNet returns two logit maps (extent head and boundary head); by default they
are combined into a single uint8 GeoTIFF where:

    0 = no-data / background
    1 = field extent
    2 = field boundary  (boundary label can take priority over extent via cli parameter)

With ``--save_scores`` the raw stacked logits (extent channels first, boundary
channels second) are written as float32 instead.

The model expects a 3-channel RGB image.  If the input is an 8-band bi-temporal
acquisition (e.g. RGBNIR x 2 windows), use ``--temporal t0`` (bands 1-3) or
``--temporal t1`` (bands 5-7) to extract the desired window.

Usage example:
    uv run scripts/kuva/rgb_inference_baseline.py \\
    /path/to/input.tif \\ 
    --model /path/to/faunet.pth \\
    --out /path/to/output.tif \\
    --temporal t0 \\ 
    --gpu 0 \\ 
    --patch_size 256 \\ 
    --padding 0 \\ 
    --overwrite \\ 
    --no-boundary-priority \\ 
"""

import contextlib
import os
import sys
from typing import Callable

import click
import kornia.augmentation as K
import numpy as np
import rasterio
import torch
from kornia.constants import Resample
from rasterio.enums import ColorInterp
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from torchgeo.samplers import GridGeoSampler
from tqdm import tqdm

import kuva.models.faunet as _faunet_mod
from ftw_tools.inference.inference import SingleRasterDataset, setup_inference
from kuva.models.faunet import FAUNet, faunet_normalize


def make_preprocess_fn(temporal: str | None) -> Callable:
    """Return a sample-level transform for the torchgeo dataset.

    Args:
        temporal: ``"t0"`` → bands 0-2, ``"t1"`` → bands 4-6, ``None`` → all bands
            (input must already be 3-band).

    Normalisation uses ``faunet_normalize``: per-channel min-max with a 2000 clip,
    matching the preprocessing used during FAUNet training.
    """
    if temporal == "t0":
        band_indices = [0, 1, 2]
    elif temporal == "t1":
        band_indices = [4, 5, 6]
    else:
        band_indices = None  # identity selection

    def preprocess(sample: dict) -> dict:
        img = sample["image"]  # (C, H, W) float32 tensor from torchgeo
        if band_indices is not None:
            img = img[band_indices, :, :]
        # faunet_normalize expects (C, H, W) numpy array with CHW=True
        normalized = faunet_normalize(img.numpy(), CHW=True)
        sample["image"] = torch.from_numpy(normalized).float()
        return sample

    return preprocess


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _faunet_as_main():
    """Temporarily expose the faunet module as __main__ so pickle can resolve its classes."""
    sys.modules["__main__"], _orig = _faunet_mod, sys.modules["__main__"]
    try:
        yield
    finally:
        sys.modules["__main__"] = _orig


def load_faunet(checkpoint_path: str, device: torch.device) -> FAUNet:
    with _faunet_as_main():
        return torch.load(
            checkpoint_path, map_location=device, weights_only=False
        ).eval()


def run_faunet_inference(
    input: str,
    model_path: str,
    out: str,
    temporal: str | None,
    n_classes: int,
    resize_factor: int,
    gpu: int | None,
    patch_size: int | None,
    batch_size: int,
    num_workers: int,
    padding: int | None,
    overwrite: bool,
    mps_mode: bool,
    save_scores: bool,
    boundary_priority: bool = True,
) -> None:
    device, transform, input_shape, patch_size, stride, padding = setup_inference(
        input, out, gpu, patch_size, padding, overwrite, mps_mode
    )

    # ---- Model ----
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    model = load_faunet(model_path, device=device)

    # ---- Up/down-samplers (for resize_factor) ----
    sam_device = "cpu" if mps_mode else device
    up_sample = K.Resize((patch_size * resize_factor,) * 2).to(sam_device)
    down_sample = K.Resize((patch_size,) * 2, resample=Resample.NEAREST.name).to(
        sam_device
    )

    # ---- Dataset / DataLoader ----
    preprocess_fn = make_preprocess_fn(temporal)
    dataset = SingleRasterDataset(input, transforms=preprocess_fn)
    sampler = GridGeoSampler(dataset, size=patch_size, stride=stride)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=stack_samples,
    )

    # ---- Output array ----
    input_height, input_width = input_shape[0], input_shape[1]
    if save_scores:
        out_channels = 2 * n_classes  # extent channels + boundary channels stacked
        output_mask = np.zeros(
            [out_channels, input_height, input_width], dtype=np.float32
        )
    else:
        out_channels = 1
        output_mask = np.zeros([1, input_height, input_width], dtype=np.uint8)

    # ---- Inference loop ----
    for batch in tqdm(dataloader, desc="Running inference"):
        images = batch["image"]  # (B, C, H, W)
        images = up_sample(images)

        if not mps_mode:
            images = images.to(device)

        bboxes = [
            (b[0].item(), b[3].item(), b[1].item(), b[4].item())
            for b in batch["bounds"]
        ]

        with torch.inference_mode():
            logits, logits_edge = model(images)  # each (B, n_classes, H, W)

            if save_scores:
                # Stack extent and boundary logits along channel axis
                scores = torch.cat(
                    [logits, logits_edge], dim=1
                )  # (B, 2*n_classes, H, W)
                scores = down_sample(scores.float())
                predictions = scores.cpu().numpy()  # float32
            else:
                # Argmax each head independently
                extent = logits.argmax(dim=1)  # (B, H, W)  0=bg, 1=field
                boundary = logits_edge.argmax(dim=1)  # (B, H, W)  0=bg, 1=boundary

                # Combine: field=1, boundary=2
                if boundary_priority:
                    combined = torch.where(
                        boundary == 1, 2, torch.where(extent == 1, 1, 0)
                    )
                else:
                    combined = torch.where(
                        extent == 1, 1, torch.where(boundary == 1, 2, 0)
                    )
                combined = combined.to(torch.uint8)

                combined = combined.unsqueeze(1).float()  # (B, 1, H, W)
                combined = down_sample(combined).byte()
                predictions = combined.cpu().numpy()  # uint8 (B, 1, H, W)

        # ---- Stitch patches into output array ----
        for i in range(len(bboxes)):
            minx, miny, maxx, maxy = bboxes[i]

            left, bottom = ~transform * (minx, miny)
            right, top = ~transform * (maxx, maxy)
            left, right, top, bottom = (
                int(np.round(left)),
                int(np.round(right)),
                int(np.round(top)),
                int(np.round(bottom)),
            )

            # Per-side effective padding (no padding when on image border)
            effective_left_pad = 0 if left <= 0 else padding
            effective_right_pad = 0 if right >= input_width else padding
            effective_top_pad = 0 if top <= 0 else padding
            effective_bottom_pad = 0 if bottom >= input_height else padding

            # Interior region (after trimming padding) in destination coordinates
            pleft = left + effective_left_pad
            pright = right - effective_right_pad
            ptop = top + effective_top_pad
            pbottom = bottom - effective_bottom_pad

            # Clamp to image bounds
            dst_left = max(pleft, 0)
            dst_top = max(ptop, 0)
            dst_right = min(pright, input_width)
            dst_bottom = min(pbottom, input_height)

            # Corresponding source indices within the prediction patch
            src_left = effective_left_pad + (dst_left - pleft)
            src_right = effective_left_pad + (dst_right - pleft)
            src_top = effective_top_pad + (dst_top - ptop)
            src_bottom = effective_top_pad + (dst_bottom - ptop)

            _, h, w = predictions[i].shape
            src_left = max(0, min(src_left, w))
            src_right = max(0, min(src_right, w))
            src_top = max(0, min(src_top, h))
            src_bottom = max(0, min(src_bottom, h))

            if src_right <= src_left or src_bottom <= src_top:
                continue

            inp = predictions[i, :, src_top:src_bottom, src_left:src_right]
            output_mask[:, dst_top:dst_bottom, dst_left:dst_right] = inp

    # ---- Save output ----
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with rasterio.open(input) as src:
        profile = src.profile
        tags = src.tags()

    profile.update(
        {
            "driver": "GTiff",
            "count": out_channels,
            "dtype": "float32" if save_scores else "uint8",
            "compress": "lzw",
            "predictor": 2,
            "blockxsize": 512,
            "blockysize": 512,
            "tiled": True,
            "interleave": "pixel",
            "nodata": None if save_scores else 0,
        }
    )

    with rasterio.open(out, "w", **profile) as dst:
        dst.update_tags(**tags)
        if save_scores:
            dst.write(output_mask)
        else:
            dst.write_colormap(1, {1: (255, 0, 0), 2: (0, 255, 0)})
            dst.colorinterp = [ColorInterp.palette]
            dst.write(output_mask[0], 1)

    print(f"\nSaved output to: {out}")


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--model",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the FAUNet checkpoint (.pth).",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path for the output GeoTIFF.",
)
@click.option(
    "--temporal",
    type=click.Choice(["t0", "t1"]),
    default=None,
    show_default=True,
    help=(
        "Band selection for 8-band bi-temporal inputs.  "
        "'t0' uses bands 1-3 (BGR/RGB first window), "
        "'t1' uses bands 5-7 (second window).  "
        "Omit for 3-band inputs."
    ),
)
@click.option(
    "--n_classes",
    type=int,
    default=2,
    show_default=True,
    help="Number of output classes per FAUNet head.",
)
@click.option(
    "--resize_factor",
    type=int,
    default=1,
    show_default=True,
    help="Resize factor applied to patches before inference.",
)
@click.option(
    "--gpu", type=int, default=None, help="CUDA device index.  Omit to use CPU."
)
@click.option(
    "--patch_size",
    type=int,
    default=None,
    help="Patch size in pixels.  Auto-detected when omitted.",
)
@click.option(
    "--padding",
    type=int,
    default=None,
    help="Padding in pixels.  Auto-selected when omitted.",
)
@click.option(
    "--batch_size",
    type=int,
    default=4,
    show_default=True,
    help="DataLoader batch size.",
)
@click.option(
    "--num_workers",
    type=int,
    default=4,
    show_default=True,
    help="DataLoader worker processes.",
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing output file."
)
@click.option(
    "--mps_mode", is_flag=True, default=False, help="Use Apple MPS backend (macOS)."
)
@click.option(
    "--save_scores",
    is_flag=True,
    default=False,
    help=(
        "Save raw stacked logits (extent + boundary, float32) instead of argmax labels."
    ),
)
@click.option(
    "--boundary_priority/--no-boundary-priority",
    default=True,
    show_default=True,
    help="When a pixel is both field and boundary, boundary wins (default) or field wins.",
)
def main(
    input: str,
    model: str,
    out: str,
    temporal: str | None,
    n_classes: int,
    resize_factor: int,
    gpu: int | None,
    patch_size: int | None,
    padding: int | None,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    mps_mode: bool,
    save_scores: bool,
    boundary_priority: bool,
) -> None:
    """Run FAUNet inference on INPUT and write a labelled GeoTIFF to --out.

    OUTPUT LABELS (default mode):

    \b
        0  background / no-data
        1  field extent
        2  field boundary  (takes priority over extent when both predict positive)

    With --save_scores: writes 2*n_classes float32 bands (extent head channels
    followed by boundary head channels) without any remapping.
    """
    run_faunet_inference(
        input=input,
        model_path=model,
        out=out,
        temporal=temporal,
        n_classes=n_classes,
        resize_factor=resize_factor,
        gpu=gpu,
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        padding=padding,
        overwrite=overwrite,
        mps_mode=mps_mode,
        save_scores=save_scores,
        boundary_priority=boundary_priority,
    )


if __name__ == "__main__":
    main()

"""Batch inference script: runs inference for multiple models on a single input image.

Usage example:
    python scripts/kuva/batch_inference.py /path/to/input.tif \
        --models /path/to/model1.ckpt /path/to/model2.ckpt \
        --out_dir /path/to/output_dir \
        --resize_factor 1 \
        --gpu 0 \
        --batch_size 8 \
        --num_workers 8
"""

import argparse
import os
from pathlib import Path

from ftw_tools.inference.inference import run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference for multiple model checkpoints on a single input image."
    )
    parser.add_argument("input", type=str, help="Path to the input .tif file.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model checkpoint paths or registry model names.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory where inference results will be saved.",
    )
    parser.add_argument(
        "--resize_factor",
        type=int,
        default=2,
        help="Resize factor for inference (default: 2).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device index to use. If not set, CPU is used.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4).",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Patch size in pixels. Auto-selected if not provided.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=None,
        help="Padding in pixels. Auto-selected if not provided.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--mps_mode",
        action="store_true",
        help="Use Apple MPS backend instead of CUDA.",
    )
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="Save raw softmax scores instead of argmax labels.",
    )
    return parser.parse_args()


def model_stem(model: str) -> str:
    """Return a filesystem-safe stem for naming the output file."""
    return Path(model).stem if model.endswith(".ckpt") else model


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    input_stem = Path(args.input).stem

    for model in args.models:
        stem = model_stem(model)
        out_filename = f"{input_stem}_{stem}.tif"
        out_path = os.path.join(args.out_dir, out_filename)

        print(f"\n=== Running inference with model: {model} ===")
        print(f"Output: {out_path}")

        run(
            input=args.input,
            model=model,
            out=out_path,
            resize_factor=args.resize_factor,
            gpu=args.gpu,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            padding=args.padding,
            overwrite=args.overwrite,
            mps_mode=args.mps_mode,
            save_scores=args.save_scores,
        )

    print("\nAll models finished.")


if __name__ == "__main__":
    main()

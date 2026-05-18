"""Run all patch_size / resize_factor / padding combinations on a single input image.

Suffixes in output filenames encode the parameters:
    ps<patch_size>_rf<resize_factor>_pad<padding>

Usage:
    python scripts/kuva/run_sampling_combinations.py /path/to/input.tif \
        --model /path/to/model.ckpt \
        --out_dir /path/to/output_dir \
        --gpu 7 \
        --batch_size 8 \
        --num_workers 8
"""

import argparse
import gc
import os
import time
from pathlib import Path

import torch

from ftw_tools.inference.inference import run

# (patch_size, resize_factor, padding, label)
# padding=None means the default auto-calculated value (patch_size // 16)
COMBINATIONS = [
    # --- Baseline: matches training distribution exactly ---
    (256, 2, 16, "baseline"),
    # --- Undersample: larger patch → more ground coverage per call ---
    (512, 2, 32, "undersample_ps512"),
    (1024, 2, 64, "undersample_ps1024"),
    # --- Oversample: smaller patch → finer spatial resolution ---
    (128, 2, 8, "oversample_ps128"),
    # --- resize_factor variants (patch_size fixed at 256) ---
    (256, 1, 16, "rf1_baseline_ps256"),  # model sees 256×256 (below training size)
    (256, 4, 16, "rf4_baseline_ps256"),  # model sees 1024×1024 (high memory)
    # --- Overlap / stitching variants (patch_size=256, resize_factor=2) ---
    (256, 2, 64, "overlap50pct"),  # stride=128, ~50% overlap
    (256, 2, 88, "overlap69pct"),  # stride=80, ~69% overlap
    # --- Same model input (512×512) as training, different ground coverage ---
    (512, 1, 32, "same_input_undersample"),   # model sees 512×512, 2× ground coverage
    (128, 4, 8,  "same_input_oversample"),    # model sees 512×512, 0.5× ground coverage
    # --- High overlap for larger patch sizes ---
    (512, 2, 128, "undersample_ps512_overlap50pct"),   # stride=256, 50% overlap
    (1024, 2, 192, "undersample_ps1024_overlap37pct"), # stride=640, ~37% overlap
    # --- Intermediate resize factor ---
    (256, 3, 16, "rf3_baseline_ps256"),  # model sees 768×768
    # --- Large patch, no upsampling (maximum ground coverage, low VRAM) ---
    (1024, 1, 64, "undersample_ps1024_rf1"),  # model sees 1024×1024, widest context
]


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling combination experiments for a single model."
    )
    parser.add_argument("input", type=str, help="Path to the input .tif file.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .ckpt checkpoint or registry model name.",
    )
    parser.add_argument(
        "--gpu", type=int, default=7, help="GPU device index to use (default: 7)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers (default: 8).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/sampling_combinations",
        help="Output directory (default: output/sampling_combinations).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    input_stem = Path(args.input).stem

    stats = []

    for patch_size, resize_factor, padding, label in COMBINATIONS:
        suffix = f"ps{patch_size}_rf{resize_factor}_pad{padding}_{label}"
        out_filename = f"{input_stem}_{suffix}.tif"
        out_path = os.path.join(args.out_dir, out_filename)

        print(f"\n{'=' * 60}")
        print(f"  Combo : {label}")
        print(
            f"  patch_size={patch_size}  resize_factor={resize_factor}  padding={padding}"
        )
        print(f"  Output: {out_path}")
        print(f"{'=' * 60}")

        if os.path.exists(out_path) and not args.overwrite:
            print("  Skipping (already exists). Use --overwrite to force.")
            stats.append(
                {
                    "label": label,
                    "patch_size": patch_size,
                    "resize_factor": resize_factor,
                    "padding": padding,
                    "elapsed_s": None,
                    "peak_vram_mb": None,
                    "skipped": True,
                }
            )
            continue

        if torch.cuda.is_available() and args.gpu >= 0:
            torch.cuda.set_device(args.gpu)
            torch.cuda.reset_peak_memory_stats(args.gpu)

        t0 = time.time()
        run(
            input=args.input,
            model=args.model,
            out=out_path,
            resize_factor=resize_factor,
            gpu=args.gpu,
            patch_size=patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            padding=padding,
            overwrite=args.overwrite,
            mps_mode=False,
            save_scores=False,
        )
        elapsed = time.time() - t0

        peak_vram_mb = None
        if torch.cuda.is_available() and args.gpu >= 0:
            peak_vram_mb = torch.cuda.max_memory_allocated(args.gpu) / 1024**2
            gc.collect()
            torch.cuda.empty_cache()

        stats.append(
            {
                "label": label,
                "patch_size": patch_size,
                "resize_factor": resize_factor,
                "padding": padding,
                "elapsed_s": elapsed,
                "peak_vram_mb": peak_vram_mb,
                "skipped": False,
            }
        )

    # Print summary table
    print(f"\n{'=' * 75}")
    print(
        f"  {'Label':<25} {'ps':>5} {'rf':>4} {'pad':>5} {'time (s)':>10} {'VRAM (MB)':>11}"
    )
    print(f"  {'-' * 25} {'-' * 5} {'-' * 4} {'-' * 5} {'-' * 10} {'-' * 11}")
    for s in stats:
        if s["skipped"]:
            time_str = "skipped"
            vram_str = "-"
        else:
            time_str = f"{s['elapsed_s']:.1f}"
            vram_str = (
                f"{s['peak_vram_mb']:.0f}" if s["peak_vram_mb"] is not None else "N/A"
            )
        print(
            f"  {s['label']:<25} {s['patch_size']:>5} {s['resize_factor']:>4} "
            f"{s['padding']:>5} {time_str:>10} {vram_str:>11}"
        )
    print(f"{'=' * 75}")
    print("\nAll combinations finished.")


if __name__ == "__main__":
    main()

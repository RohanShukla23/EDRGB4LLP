#!/usr/bin/env python3
import os
import io
import json
import math
import argparse
from pathlib import Path

import yaml
import numpy as np
from PIL import Image

import tonic
from tonic import transforms
import h5py  # noqa: F401  # ensures hdf5plugin is loaded under the hood
import hdf5plugin  # noqa: F401

# ---------- helpers ----------

def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_npz(path: Path, array_name: str, arr: np.ndarray):
    # atomic-ish write
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, **{array_name: arr})
    os.replace(tmp, path)

def save_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def pil_resize_keep_rgb(img: Image.Image, size_hw):
    H, W = size_hw
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize((W, H), Image.BILINEAR)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--max-samples", type=int, default=50, help="process only N samples for a smoke test")
    ap.add_argument("--seq-id", default=None, help="override sequence folder name (default dsec_<split>)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    H = int(cfg["resize"]["H"])
    W = int(cfg["resize"]["W"])
    bins = int(cfg["bins"])
    window_ms = float(cfg["window_ms"])
    bucket_root = Path(cfg["bucket_root"])
    raw_root = Path(cfg["raw_local_root"])

    seq_id = args.seq_id or f"dsec_{args.split}"
    seq_root = bucket_root / "sequences" / seq_id
    rgb_dir   = ensure_dir(seq_root / "rgb")
    evt_dir   = ensure_dir(seq_root / "events")
    lab_dir   = ensure_dir(seq_root / "labels")
    meta_dir  = ensure_dir(seq_root / "meta")
    splits_dir= ensure_dir(bucket_root / "splits")

    # Build a DSEC dataset that returns: (events_left, images_rectified_left, image_timestamps)
    print("[info] creating DSEC dataset via tonic (this may download; large!)")
    ds = tonic.datasets.DSEC(
        save_to=str(raw_root),
        split=args.split,
        data_selection=[cfg["dsec_events"], cfg["dsec_camera"], "image_timestamps"],
    )

    # Voxelizer matches sensor size from dataset metadata
    voxelizer = transforms.ToVoxelGrid(
        sensor_size=ds.sensor_size,  # (H_raw, W_raw, 2)
        n_time_bins=bins
    )

    print(f"[info] dataset ready: {len(ds)} samples; target size=({H},{W}); bins={bins}; window_ms={window_ms}")

    # Process a small subset first (smoke test)
    N = len(ds) if args.max_samples <= 0 else min(args.max_samples, len(ds))
    index_width = int(math.log10(max(1, N))) + 5  # zero pad nicely

    split_records = []

    for i in range(N):
        try:
            data, _ = ds[i]  # DSEC returns (data_tuple, target_tuple_or_none)
        except Exception as e:
            print(f"[warn] skipping index {i} due to read error: {e}")
            continue

        # Expect data = (events, image, timestamp)
        if not isinstance(data, (tuple, list)) or len(data) < 3:
            print(f"[warn] unexpected data structure at {i}, got type {type(data)}; skipping")
            continue

        events, image, t_rgb = data[0], data[1], data[2]

        # events: numpy structured array with fields x,y,t,pol
        # image: numpy array (H,W,3) or (H,W) grayscale. Convert to PIL for resize.
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            print(f"[warn] unsupported image type at {i}: {type(image)}; skipping")
            continue

        # Build a voxel grid from events.
        # NOTE: In this MVP we trust Tonic's sample-level events to be aligned with this image.
        # If you later enforce a custom trailing window [t_rgb - window_ms, t_rgb],
        # you can filter by timestamps before voxelization.
        try:
            voxel = voxelizer(events)  # shape (bins, H_raw, W_raw)
        except Exception as e:
            print(f"[warn] voxelization failed at {i}: {e}; skipping")
            continue

        # Resize RGB to (H,W)
        img_resized = pil_resize_keep_rgb(img, (H, W))

        # For voxel grids, you can resize each bin with PIL via per-slice Image.fromarray
        # but simplest is to save raw voxel at sensor size; most models can handle resize on-the-fly.
        # Here we DO resize to keep pair strictly aligned at the same size.
        voxel_resized = np.stack([
            np.array(Image.fromarray(voxel[b]).resize((W, H), Image.BILINEAR))
            for b in range(voxel.shape[0])
        ], axis=0).astype(np.float32)

        # Normalize events if you like (left as-is per config)
        # Save artifacts
        frame_id = str(i).zfill(index_width)
        rgb_path   = rgb_dir  / f"{frame_id}.jpg"
        voxel_path = evt_dir  / f"{frame_id}_voxel.npz"
        label_path = lab_dir  / f"{frame_id}.json"
        meta_path  = meta_dir / f"{frame_id}.json"

        # JPEG save
        img_resized.save(rgb_path, format="JPEG", quality=95)

        # NPZ save
        save_npz(voxel_path, "voxel", voxel_resized)

        # Minimal labels placeholder (no boxes yet)
        save_json(label_path, {"boxes": [], "classes": []})

        # Meta with timestamps; we set an assumed trailing window for transparency
        t_evt_end = float(t_rgb) if hasattr(t_rgb, "__float__") else float(t_rgb[0]) if isinstance(t_rgb, (np.ndarray, list, tuple)) else None
        t_evt_start = (t_evt_end - window_ms/1000.0) if t_evt_end is not None else None
        save_json(meta_path, {
            "t_rgb": t_evt_end,
            "t_evt_start": t_evt_start,
            "t_evt_end": t_evt_end,
            "resize": {"H": H, "W": W},
            "bins": bins,
            "source": {"dataset": "DSEC", "split": args.split}
        })

        # Append to split index (relative paths under processed/v1)
        rel = seq_root.relative_to(bucket_root)
        split_records.append({
            "seq_id": seq_id,
            "frame_id": frame_id,
            "rgb": str(rel / "rgb" / f"{frame_id}.jpg"),
            "voxel": str(rel / "events" / f"{frame_id}_voxel.npz"),
            "labels_json": str(rel / "labels" / f"{frame_id}.json"),
            "meta": str(rel / "meta" / f"{frame_id}.json"),
            "H": H, "W": W, "bins": bins, "window_ms": window_ms
        })

        if (i + 1) % 10 == 0 or (i + 1) == N:
            print(f"[info] processed {i+1}/{N}")

    # Write split file (append or replace train.jsonl for this smoke test)
    split_file = splits_dir / f"{args.split}.jsonl"
    with open(split_file, "w") as f:
        for rec in split_records:
            f.write(json.dumps(rec) + "\n")

    print(f"[done] wrote {len(split_records)} records to {split_file}")
    print(f"[hint] sample rgb:   {rgb_dir}/00000{'0'*(index_width-5)}.jpg (if >= 1 sample)")
    print(f"[hint] sample voxel: {evt_dir}/00000{'0'*(index_width-5)}_voxel.npz (if >= 1 sample)")

if __name__ == "__main__":
    main()

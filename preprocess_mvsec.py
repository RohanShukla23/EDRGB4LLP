#!/usr/bin/env python3
"""
MVSEC Dataset Preprocessing Script
===================================
Processes the MVSEC (Multi Vehicle Stereo Event Camera) dataset into
a format suitable for training event-RGB fusion models.

MVSEC provides 4 scenes: outdoor_night, outdoor_day, indoor_flying, motorcycle
Each sample contains synchronized stereo event cameras, RGB cameras, IMU, and ground truth.
"""

import os
import json
import math
import argparse
from pathlib import Path

import yaml
import numpy as np
from PIL import Image
import cv2

import tonic
from tonic import transforms

# ---------- helpers ----------

def load_config(path: Path):
    """Load the YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    """Create a directory if it doesn't exist and return the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_npz(path: Path, array_name: str, arr: np.ndarray):
    """Atomically save a numpy array in compressed .npz format.
    
    This version accounts for numpy's automatic .npz extension addition.
    """
    path = path.resolve()
    tmp = path.with_suffix('.tmp')
    tmp_with_npz = Path(str(tmp) + '.npz')
    
    try:
        np.savez_compressed(tmp, **{array_name: arr})
        
        if not tmp_with_npz.exists():
            raise FileNotFoundError(f"Temporary file was not created: {tmp_with_npz}")
        
        if path.exists():
            path.unlink()
        
        tmp_with_npz.rename(path)
        
    except Exception as e:
        if tmp_with_npz.exists():
            tmp_with_npz.unlink()
        raise Exception(f"Failed to save NPZ file to {path}: {e}") from e

def save_json(path: Path, obj: dict):
    """Atomically save a JSON file."""
    path = path.resolve()
    tmp = path.with_suffix('.tmp')
    
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        
        if not tmp.exists():
            raise FileNotFoundError(f"Temporary file was not created: {tmp}")
        
        if path.exists():
            path.unlink()
        
        tmp.rename(path)
        
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise Exception(f"Failed to save JSON file to {path}: {e}") from e

def pil_resize_keep_rgb(img: Image.Image, size_hw):
    """Resize a PIL image to the target size, ensuring RGB mode."""
    H, W = size_hw
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize((W, H), Image.BILINEAR)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description='MVSEC Dataset Preprocessing')
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--scene", required=True, 
                    choices=["outdoor_night", "outdoor_day", "indoor_flying", "motorcycle"],
                    help="MVSEC scene to process")
    ap.add_argument("--max-samples", type=int, default=-1, 
                    help="process only N samples for testing (-1 for all)")
    ap.add_argument("--camera", default="left", choices=["left", "right"],
                    help="which camera to use (left or right)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    # Target resolution for resizing
    H = int(cfg["resize"]["H"])
    W = int(cfg["resize"]["W"])

    # Number of temporal bins for voxel grid
    bins = int(cfg["bins"])

    # Note: MVSEC doesn't have explicit temporal windows like DSEC
    # Events are already temporally aligned with each frame
    window_ms = float(cfg.get("window_ms", 10.0))
    
    # Output directory
    bucket_root = Path(cfg["bucket_root"])
    raw_root = Path(cfg["raw_local_root"])

    # Output directory format - include scene name
    seq_id = f"mvsec_{args.scene}"
    seq_root = bucket_root / "sequences" / seq_id
    rgb_dir   = ensure_dir(seq_root / "rgb")
    evt_dir   = ensure_dir(seq_root / "events")
    lab_dir   = ensure_dir(seq_root / "labels")
    meta_dir  = ensure_dir(seq_root / "meta")
    splits_dir= ensure_dir(bucket_root / "splits")

    # Build MVSEC dataset
    print(f"[info] loading MVSEC dataset scene: {args.scene}")
    print("[info] this may download data if not already cached")
    
    ds = tonic.datasets.MVSEC(
        save_to=str(raw_root),
        scene=args.scene,
    )

    # MVSEC sensor size is typically (346, 260, 2) for DAVIS346
    # Initialize voxelizer
    voxelizer = transforms.ToVoxelGrid(
        sensor_size=ds.sensor_size,
        n_time_bins=bins
    )

    print(f"[info] dataset ready: {len(ds)} samples")
    print(f"[info] scene: {args.scene}")
    print(f"[info] sensor size: {ds.sensor_size}")
    print(f"[info] target size=({H},{W}); bins={bins}")

    # Determine how many samples to process
    num_samples = len(ds) if args.max_samples <= 0 else min(args.max_samples, len(ds))
    
    split_records = []
    samples_processed = 0
    samples_skipped = 0

    # Determine which camera indices to use based on args
    if args.camera == "left":
        events_idx = 0  # events_left
        images_idx = 4  # images_left
        camera_name = "left"
    else:
        events_idx = 1  # events_right
        images_idx = 5  # images_right
        camera_name = "right"

    print(f"[info] using {camera_name} camera")
    print(f"[info] processing {num_samples} samples...\n")

    # Process each sample in the dataset
    for sample_idx in range(num_samples):
        try:
            # MVSEC returns (data_tuple, targets_tuple)
            # data_tuple contains: (events_left, events_right, imu_left, imu_right, 
            #                       images_left, images_right)
            # targets_tuple contains: (depth_rect_left, depth_rect_right, pose)
            data_tuple, targets_tuple = ds[sample_idx]
            
        except Exception as e:
            print(f"[warn] skipping sample {sample_idx} due to error: {e}")
            samples_skipped += 1
            continue

        # Extract events and image for the selected camera
        try:
            events = data_tuple[events_idx]  # Structured array with x, y, t, p
            image = data_tuple[images_idx]    # RGB image array
            
            # Extract depth and pose if available (optional, for future use)
            # depth = targets_tuple[events_idx] if targets_tuple else None
            # pose = targets_tuple[2] if targets_tuple and len(targets_tuple) > 2 else None
            
        except (IndexError, TypeError) as e:
            print(f"[warn] sample {sample_idx}: data structure error: {e}, skipping")
            samples_skipped += 1
            continue

        # Skip if no events
        if events is None or len(events) == 0:
            samples_skipped += 1
            continue

        # Convert image to PIL for resizing
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            print(f"[warn] sample {sample_idx}: unsupported image type: {type(image)}, skipping")
            samples_skipped += 1
            continue

        # Build voxel grid from events
        try:
            voxel = voxelizer(events)
            # Expected shape: (bins, polarity, H_sensor, W_sensor)
            # For MVSEC DAVIS346: (4, 1, 260, 346)
        except Exception as e:
            print(f"[warn] sample {sample_idx}: voxelization failed: {e}, skipping")
            samples_skipped += 1
            continue

        # Resize RGB image to target resolution
        img_resized = pil_resize_keep_rgb(img, (H, W))

        # Resize voxel grid to target resolution
        # Handle the polarity dimension: voxel[b, 0] gives us (H, W)
        try:
            voxel_resized = np.stack([
                cv2.resize(voxel[b, 0], (W, H), interpolation=cv2.INTER_LINEAR)
                for b in range(voxel.shape[0])
            ], axis=0).astype(np.float32)
            # Final shape: (bins, H_target, W_target) = (4, 320, 320)
        except Exception as e:
            print(f"[warn] sample {sample_idx}: voxel resize failed: {e}, skipping")
            samples_skipped += 1
            continue

        # Create frame ID with zero-padded index
        index_width = int(math.log10(max(1, num_samples))) + 5
        frame_id = f"{args.scene}_{camera_name}_{sample_idx:0{index_width}d}"
        
        # Define output paths
        rgb_path   = rgb_dir  / f"{frame_id}.jpg"
        voxel_path = evt_dir  / f"{frame_id}_voxel.npz"
        label_path = lab_dir  / f"{frame_id}.json"
        meta_path  = meta_dir / f"{frame_id}.json"

        # Save RGB image as JPEG
        img_resized.save(rgb_path, format="JPEG", quality=95)

        # Save voxel grid as compressed numpy array
        save_npz(voxel_path, "voxel", voxel_resized)

        # Save placeholder labels (depth and pose could be added here in the future)
        save_json(label_path, {"boxes": [], "classes": []})

        # Save metadata
        # Note: MVSEC events are already aligned with frames, no explicit timestamps
        save_json(meta_path, {
            "scene": args.scene,
            "camera": camera_name,
            "sample_index": sample_idx,
            "num_events": int(len(events)),
            "resize": {"H": H, "W": W},
            "bins": bins,
            "source": {"dataset": "MVSEC"}
        })

        # Add to split index
        rel = seq_root.relative_to(bucket_root)
        split_records.append({
            "seq_id": seq_id,
            "frame_id": frame_id,
            "scene": args.scene,
            "camera": camera_name,
            "rgb": str(rel / "rgb" / f"{frame_id}.jpg"),
            "voxel": str(rel / "events" / f"{frame_id}_voxel.npz"),
            "labels_json": str(rel / "labels" / f"{frame_id}.json"),
            "meta": str(rel / "meta" / f"{frame_id}.json"),
            "H": H, "W": W, "bins": bins
        })

        samples_processed += 1

        # Progress update
        if (sample_idx + 1) % 50 == 0 or (sample_idx + 1) == num_samples:
            print(f"[info] processed {sample_idx+1}/{num_samples} samples "
                  f"(saved: {samples_processed}, skipped: {samples_skipped})")

    # Write split file
    split_file = splits_dir / f"{args.scene}_{camera_name}.jsonl"
    with open(split_file, "w") as f:
        for rec in split_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n[done] wrote {len(split_records)} records to {split_file}")
    print(f"[info] successfully processed {samples_processed} samples")
    print(f"[info] skipped {samples_skipped} samples due to errors")

if __name__ == "__main__":
    main()
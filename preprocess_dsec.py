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
import cv2

import tonic
from tonic import transforms
import h5py  # noqa: F401  # ensures hdf5plugin is loaded under the hood
import hdf5plugin  # noqa: F401

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
    We create a temp file without .npz, numpy adds .npz, then we rename.
    """
    # Convert to absolute path to avoid Windows path issues
    path = path.resolve()
    
    # For the temporary file, we use .tmp instead of .npz
    # np.savez_compressed will automatically add .npz, creating path.tmp.npz
    # Then we rename path.tmp.npz to path.npz
    tmp = path.with_suffix('.tmp')  # Changes 'file.npz' to 'file.tmp'
    tmp_with_npz = Path(str(tmp) + '.npz')  # This is what numpy will actually create
    
    try:
        # Save to temporary file
        # This will create a file at tmp_with_npz (e.g., 'file.tmp.npz')
        np.savez_compressed(tmp, **{array_name: arr})
        
        # Verify the temporary file was created successfully
        if not tmp_with_npz.exists():
            raise FileNotFoundError(f"Temporary file was not created: {tmp_with_npz}")
        
        # On Windows, if the destination exists, we need to remove it first
        if path.exists():
            path.unlink()
        
        # Now move the temporary file to the final location
        tmp_with_npz.rename(path)
        
    except Exception as e:
        # Clean up temporary file if something went wrong
        if tmp_with_npz.exists():
            tmp_with_npz.unlink()
        raise Exception(f"Failed to save NPZ file to {path}: {e}") from e
    
def save_json(path: Path, obj: dict):
    """Atomically save a JSON file.
    
    This version is Windows-compatible and uses absolute paths to avoid
    path resolution issues on different operating systems.
    """
    # Convert to absolute path to avoid Windows path issues
    path = path.resolve()
    tmp = path.with_suffix(path.suffix + ".tmp")
    
    try:
        # Write to temporary file
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        
        # Verify the temporary file was created successfully
        if not tmp.exists():
            raise FileNotFoundError(f"Temporary file was not created: {tmp}")
        
        # On Windows, remove destination if it exists
        if path.exists():
            path.unlink()
        
        # Move temporary file to final location
        tmp.rename(path)
        
    except Exception as e:
        # Clean up temporary file if something went wrong
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--max-samples", type=int, default=50, 
                    help="process only N recordings (not frames) for testing")
    ap.add_argument("--seq-id", default=None, 
                    help="override sequence folder name (default dsec_<split>)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    # Target resolution for resizing
    H = int(cfg["resize"]["H"])
    W = int(cfg["resize"]["W"])

    # Number of temporal bins for voxel grid
    bins = int(cfg["bins"])

    # Millisecond window for event accumulation
    window_ms = float(cfg["window_ms"])
    window_us = window_ms * 1000  # Convert to microseconds for DSEC timestamps
    
    # Output directory
    bucket_root = Path(cfg["bucket_root"])

    # Where raw DSEC data is cached
    raw_root = Path(cfg["raw_local_root"])

    # Output directory format
    seq_id = args.seq_id or f"dsec_{args.split}"
    seq_root = bucket_root / "sequences" / seq_id
    rgb_dir   = ensure_dir(seq_root / "rgb")
    evt_dir   = ensure_dir(seq_root / "events")
    lab_dir   = ensure_dir(seq_root / "labels")
    meta_dir  = ensure_dir(seq_root / "meta")
    splits_dir= ensure_dir(bucket_root / "splits")

    # Build a DSEC dataset
    # Note: DSEC returns entire recording sequences, not individual frames
    print("[info] creating DSEC dataset via tonic (this may download; large!)")
    ds = tonic.datasets.DSEC(
        save_to=str(raw_root),
        split=args.split,
        data_selection=[cfg["dsec_events"], cfg["dsec_camera"], "image_timestamps"],
    )
    
    # Fix Tonic bug: recording_selection is dict_keys instead of list
    # This prevents indexing with ds[i] from working properly
    if isinstance(ds.recording_selection, type({}.keys())):
        print("[info] fixing Tonic bug: converting recording_selection to list")
        ds.recording_selection = list(ds.recording_selection)

    # Initialize the voxelizer
    # Note: DSEC sensor_size is a 3-tuple (width, height, polarity_channels)
    # The voxelizer uses this to create voxel grids with shape (bins, polarity, H, W)
    voxelizer = transforms.ToVoxelGrid(
        sensor_size=ds.sensor_size,
        n_time_bins=bins
    )

    print(f"[info] dataset ready: {len(ds)} recordings")
    print(f"[info] target size=({H},{W}); bins={bins}; window_ms={window_ms}")

    # Determine how many recordings to process
    num_recordings = len(ds) if args.max_samples <= 0 else min(args.max_samples, len(ds))
    
    split_records = []
    total_frames_processed = 0

    # Process each recording in the dataset
    # Each "sample" in DSEC is actually an entire driving sequence with hundreds of frames
    for recording_idx in range(num_recordings):
        try:
            # DSEC returns (data_list, target_list)
            # data_list contains: [events_dict, images_array, timestamps_array]
            data_list, target_list = ds[recording_idx]
        except Exception as e:
            print(f"[warn] skipping recording {recording_idx} due to error: {e}")
            continue

        # Unpack the recording-level data
        events_dict = data_list[0]  # Dict with 'events_left' and 'ms_to_idx'
        images = data_list[1]       # Array of shape (num_frames, H, W, 3)
        timestamps = data_list[2]   # Array of shape (num_frames,) in microseconds

        # Extract the events structured array and lookup table
        events = events_dict['events_left']  # Structured array with fields: x, y, t, p
        ms_to_idx = events_dict['ms_to_idx']  # Lookup table for temporal indexing
        
        num_frames = len(timestamps)
        recording_name = ds.recording_selection[recording_idx]
        
        print(f"\n[info] Processing recording {recording_idx+1}/{num_recordings}: {recording_name}")
        print(f"  {num_frames} frames, {len(events)} events")

        # Process each frame in this recording
        for frame_idx in range(num_frames):
            # Progress indicator every 50 frames
            if frame_idx % 50 == 0:
                print(f"    processing frame {frame_idx}/{num_frames}...")
            
            # Create a unique frame identifier combining recording name and frame index
            frame_id = f"{recording_name}_{frame_idx:06d}"
            
            # Check if this frame is already processed
            rgb_path   = rgb_dir  / f"{frame_id}.jpg"
            voxel_path = evt_dir  / f"{frame_id}_voxel.npz"
            label_path = lab_dir  / f"{frame_id}.json"
            meta_path  = meta_dir / f"{frame_id}.json"
            
            # Skip if all output files already exist
            if rgb_path.exists() and voxel_path.exists() and label_path.exists() and meta_path.exists():
                # Still add to split records for completeness
                rel = seq_root.relative_to(bucket_root)
                split_records.append({
                    "seq_id": seq_id,
                    "frame_id": frame_id,
                    "recording": recording_name,
                    "rgb": str(rel / "rgb" / f"{frame_id}.jpg"),
                    "voxel": str(rel / "events" / f"{frame_id}_voxel.npz"),
                    "labels_json": str(rel / "labels" / f"{frame_id}.json"),
                    "meta": str(rel / "meta" / f"{frame_id}.json"),
                    "H": H, "W": W, "bins": bins, "window_ms": window_ms
                })
                total_frames_processed += 1
                continue
            
            # Get the RGB frame for this timestamp
            image = images[frame_idx]  # Shape: (H, W, 3), dtype: uint8
            t_rgb = timestamps[frame_idx]  # Timestamp in microseconds

            # Define the temporal window for event extraction
            # We want events that occurred in the [window_ms] milliseconds before this frame
            t_start = t_rgb - window_us
            t_end = t_rgb
            
            # OPTIMIZED: Use ms_to_idx lookup instead of filtering entire array
            t_start_ms = int((t_rgb - window_us) / 1000)
            t_end_ms = int(t_rgb / 1000)
            
            try:
                # Use the lookup table for faster indexing
                start_idx = ms_to_idx.get(t_start_ms, 0)
                end_idx = ms_to_idx.get(t_end_ms, len(events))
                frame_events = events[start_idx:end_idx]
            except (KeyError, IndexError, TypeError):
                # Fallback to original filtering if lookup fails
                event_mask = (events['t'] >= t_start) & (events['t'] < t_end)
                frame_events = events[event_mask]

            # Skip frames with no events (could happen in very static scenes)
            if len(frame_events) == 0:
                continue

            # Convert numpy image to PIL for resizing
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            else:
                img = image

            # Build voxel grid from the temporal window of events
            # The voxelizer bins events across time and space to create a dense tensor
            try:
                voxel = voxelizer(frame_events)
                # Shape will be (bins, polarity, H_sensor, W_sensor)
                # For DSEC with default settings: (4, 1, 480, 640)
            except Exception as e:
                print(f"  [warn] frame {frame_idx}: voxelization failed: {e}, skipping")
                continue

            # Resize RGB image to target resolution
            img_resized = pil_resize_keep_rgb(img, (H, W))

            # Resize voxel grid to target resolution
            # Key fix: voxel has shape (bins, 1, H, W) due to polarity dimension
            # We need voxel[b, 0] to get a 2D array (H, W) that cv2.resize can handle
            voxel_resized = np.stack([
                cv2.resize(voxel[b, 0], (W, H), interpolation=cv2.INTER_LINEAR)
                for b in range(voxel.shape[0])
            ], axis=0).astype(np.float32)
            # Final shape: (bins, H_target, W_target) = (4, 320, 320)

            # Save RGB image as JPEG
            img_resized.save(rgb_path, format="JPEG", quality=95)

            # Save voxel grid as compressed numpy array
            save_npz(voxel_path, "voxel", voxel_resized)

            # Save placeholder labels (empty for now, can be populated later)
            save_json(label_path, {"boxes": [], "classes": []})

            # Save metadata with timing and provenance information
            save_json(meta_path, {
                "t_rgb": float(t_rgb / 1e6),  # Convert microseconds to seconds
                "t_evt_start": float(t_start / 1e6),
                "t_evt_end": float(t_end / 1e6),
                "num_events": int(len(frame_events)),
                "recording": recording_name,
                "frame_in_recording": frame_idx,
                "resize": {"H": H, "W": W},
                "bins": bins,
                "window_ms": window_ms,
                "source": {"dataset": "DSEC", "split": args.split}
            })

            # Add this frame to the split index
            # Store relative paths so the dataset is portable
            rel = seq_root.relative_to(bucket_root)
            split_records.append({
                "seq_id": seq_id,
                "frame_id": frame_id,
                "recording": recording_name,
                "rgb": str(rel / "rgb" / f"{frame_id}.jpg"),
                "voxel": str(rel / "events" / f"{frame_id}_voxel.npz"),
                "labels_json": str(rel / "labels" / f"{frame_id}.json"),
                "meta": str(rel / "meta" / f"{frame_id}.json"),
                "H": H, "W": W, "bins": bins, "window_ms": window_ms
            })

            total_frames_processed += 1

        # Progress update after each recording
        if (recording_idx + 1) % 5 == 0 or (recording_idx + 1) == num_recordings:
            print(f"[info] processed {recording_idx+1}/{num_recordings} recordings, "
                  f"{total_frames_processed} total frames")

    # Write the split index file
    # Each line is a JSON object describing one frame
    split_file = splits_dir / f"{args.split}.jsonl"
    with open(split_file, "w") as f:
        for rec in split_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n[done] wrote {len(split_records)} frame records to {split_file}")
    print(f"[info] processed {total_frames_processed} frames from {num_recordings} recordings")

if __name__ == "__main__":
    main()
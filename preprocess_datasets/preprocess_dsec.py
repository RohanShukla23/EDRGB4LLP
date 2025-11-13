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
import h5py
import hdf5plugin  # noqa: F401

# ---------- helpers ----------

def load_events_efficiently(events_path: Path, t_start_us: int, t_end_us: int, max_events: int = 20_000_000):
    """
    Load only the events we need from HDF5 file using binary search.
    Much faster than loading all events and filtering.
    
    Args:
        events_path: Path to events.h5 file
        t_start_us: Start timestamp in microseconds
        t_end_us: End timestamp in microseconds
        max_events: Safety limit on number of events to return
    
    Returns:
        Structured numpy array with fields: x, y, t, p
    """
    with h5py.File(events_path, 'r') as f:
        # DSEC stores events in datasets: 'events/x', 'events/y', 'events/t', 'events/p'
        t_data = f['events/t']
        total_events = len(t_data)
        
        # Binary search for start index
        left, right = 0, total_events
        while left < right:
            mid = (left + right) // 2
            if t_data[mid] < t_start_us:
                left = mid + 1
            else:
                right = mid
        start_idx = left
        
        # Binary search for end index
        left, right = start_idx, total_events
        while left < right:
            mid = (left + right) // 2
            if t_data[mid] < t_end_us:
                left = mid + 1
            else:
                right = mid
        end_idx = left
        
        # Safety check
        num_events = end_idx - start_idx
        if num_events > max_events:
            raise ValueError(f"Too many events in window: {num_events:,} (limit: {max_events:,})")
        
        if num_events == 0:
            # Return empty structured array
            return np.array([], dtype=[('x', np.int16), ('y', np.int16), ('t', np.int64), ('p', np.int16)])
        
        # Load only the slice we need
        x = f['events/x'][start_idx:end_idx]
        y = f['events/y'][start_idx:end_idx]
        t = f['events/t'][start_idx:end_idx]
        p = f['events/p'][start_idx:end_idx]
        
        # Create structured array matching Tonic's format
        events = np.zeros(num_events, dtype=[('x', np.int16), ('y', np.int16), ('t', np.int64), ('p', np.int16)])
        events['x'] = x
        events['y'] = y
        events['t'] = t
        events['p'] = p
        
        return events

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
    processed_frame_ids = set()  # Track which frames we've already recorded
    
    # Load existing split file if it exists (for resume capability)
    split_file = splits_dir / f"{args.split}.jsonl"
    if split_file.exists():
        print(f"[info] loading existing split file: {split_file}")
        with open(split_file, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                split_records.append(record)
                processed_frame_ids.add(record["frame_id"])  # Track processed frames
        print(f"[info] loaded {len(split_records)} existing frame records")

    # Process each recording in the dataset
    # Each "sample" in DSEC is actually an entire driving sequence with hundreds of frames
    for recording_idx in range(num_recordings):
        recording_name = ds.recording_selection[recording_idx]
        print(f"\n⏳ Processing recording {recording_idx+1}/{num_recordings}: {recording_name}")
        
        # Path to the events HDF5 file for this recording
        events_h5_path = raw_root / "DSEC" / recording_name / "events_left" / "events.h5"
        
        if not events_h5_path.exists():
            print(f"[ERROR] Events file not found: {events_h5_path}")
            continue
        
        print(f"   [1/3] Loading images and timestamps (lightweight)...")
        import time
        start_time = time.time()
        
        try:
            # Only load images and timestamps from Tonic (not the huge event array)
            # We'll load events on-demand from HDF5 file
            data_list, target_list = ds[recording_idx]
        except Exception as e:
            print(f"[ERROR] Failed to load recording {recording_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"   [2/3] Unpacking data structures...")
        # Unpack the recording-level data
        events_dict = data_list[0]  # Dict with 'events_left' and 'ms_to_idx'
        images = data_list[1]       # Array of shape (num_frames, H, W, 3)
        
        # Load timestamps from file (not included in data_list when using minimal data_selection)
        ts_path = raw_root / "DSEC" / recording_name / "image_timestamps" / f"{recording_name}_image_timestamps.txt"
        timestamps = np.loadtxt(ts_path, dtype=np.int64)  # Absolute timestamps in microseconds
        
        # Get timestamp offset for converting to relative timestamps (required for HDF5 lookup)
        # DSEC HDF5 files store events with timestamps relative to recording start
        t_offset = timestamps[0]  # First frame timestamp is the reference point

        # We WON'T extract the full events array - too slow!
        # events = events_dict['events_left']  # <-- DON'T DO THIS
        
        num_frames = len(timestamps)
        elapsed = time.time() - start_time
        print(f"   [3/3] Ready to process {num_frames} frames (loaded in {elapsed:.1f}s)")

        frames_skipped = 0
        frames_processed_this_recording = 0
        
        # Process each frame in this recording
        for frame_idx in range(num_frames):
            # Progress indicator every 50 frames
            if frame_idx % 50 == 0:
                elapsed_time = frames_processed_this_recording * 0.5  # rough estimate
                print(f"    frame {frame_idx}/{num_frames} | skipped: {frames_skipped} | processed: {frames_processed_this_recording}")
            
            # Create a unique frame identifier combining recording name and frame index
            frame_id = f"{recording_name}_{frame_idx:06d}"
            
            # FAST CHECK: Skip if we already have this frame in our split records
            if frame_id in processed_frame_ids:
                frames_skipped += 1
                total_frames_processed += 1
                continue
            
            # Check if this frame is already processed (file existence check)
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
                processed_frame_ids.add(frame_id)
                total_frames_processed += 1
                frames_skipped += 1
                continue
            
            # Get the RGB frame for this timestamp
            image = images[frame_idx]  # Shape: (H, W, 3), dtype: uint8
            t_rgb = timestamps[frame_idx]  # Timestamp in microseconds (absolute)

            # Define the temporal window for event extraction
            # We want events that occurred in the [window_ms] milliseconds before this frame
            t_start = t_rgb - window_us
            t_end = t_rgb
            
            # Convert to relative timestamps for HDF5 lookup
            # DSEC HDF5 files store event timestamps relative to recording start
            t_start_rel = t_start - t_offset
            t_end_rel = t_end - t_offset
            
            # OPTIMIZED: Load only the events we need directly from HDF5 file
            # This is MUCH faster than loading all 1.8B events into memory
            try:
                frame_events = load_events_efficiently(events_h5_path, t_start_rel, t_end_rel)
                # Convert timestamps back to absolute for consistency with original code
                frame_events['t'] = frame_events['t'] + t_offset
            except ValueError as e:
                # Too many events in window
                print(f"\n[ERROR] frame {frame_idx}: {e}")
                raise
            except Exception as e:
                print(f"\n[ERROR] frame {frame_idx}: failed to load events from HDF5: {e}")
                raise

            # Skip frames with no events (could happen in very static scenes)
            if len(frame_events) == 0:
                print(f"    [warn] frame {frame_idx} has no events, skipping")
                continue

            # Convert numpy image to PIL for resizing
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            else:
                img = image

            # Build voxel grid from the temporal window of events
            # The voxelizer bins events across time and space to create a dense tensor
            # Note: Safety check for event count is already done in load_events_efficiently()
            try:
                voxel = voxelizer(frame_events)
                # Shape will be (bins, polarity, H_sensor, W_sensor)
                # For DSEC with default settings: (4, 1, 480, 640)
            except MemoryError as e:
                print(f"\n[ERROR] frame {frame_idx}: out of memory during voxelization ({len(frame_events):,} events)")
                raise
            except Exception as e:
                print(f"\n[ERROR] frame {frame_idx}: voxelization failed: {e}")
                raise

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
            processed_frame_ids.add(frame_id)
            total_frames_processed += 1
            frames_processed_this_recording += 1

        # Summary after each recording
        print(f"  ✓ Recording complete: {frames_processed_this_recording} new frames processed, {frames_skipped} skipped (already exist)")
        
        # SAVE PROGRESS: Write split file after each recording completes (only if new frames were added)
        # This ensures we don't lose progress if something fails later
        if frames_processed_this_recording > 0:
            print(f"  💾 Saving progress to {split_file.name}...")
            with open(split_file, "w") as f:
                for rec in split_records:
                    f.write(json.dumps(rec) + "\n")

        # Progress update after each recording
        if (recording_idx + 1) % 5 == 0 or (recording_idx + 1) == num_recordings:
            print(f"[info] processed {recording_idx+1}/{num_recordings} recordings, "
                  f"{total_frames_processed} total frames")

    # Final write of the split index file (redundant but ensures final state is saved)
    print(f"\n[info] Writing final split file: {split_file}")
    with open(split_file, "w") as f:
        for rec in split_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n[done] wrote {len(split_records)} frame records to {split_file}")
    print(f"[info] processed {total_frames_processed} frames from {num_recordings} recordings")

if __name__ == "__main__":
    main()
# save as carla_rgb_dvs_side_by_side.py

# --- Make CARLA egg importable on Windows ---
import sys
egg = r"C:\Users\jakep\Desktop\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg"
if egg not in sys.path:
    sys.path.append(egg)
# --------------------------------------------

import os
import numpy as np
import cv2
import carla
import time
from collections import deque

# -----------------------------
# Config
# -----------------------------
IMG_W, IMG_H = 640, 384
FOV = 90
WORLD_DT = 0.01          # 100 Hz world
RGB_TICK = 0.05          # 20 Hz RGB (for viz/labels) - also video frame rate
DVS_TICK = 0.01          # 100 Hz DVS (match world)
EVENT_WINDOW = 0.05      # seconds; build DVS image from last 50ms
DRIVE_SECONDS = 10.0     # capture duration in seconds
VIDEO_FPS = 20           # output video frame rate (matches RGB_TICK)

# DVS thresholds (safe defaults; adjust later)
POS_TH = "0.3"
NEG_TH = "0.3"
REFRACTORY_NS = "0"      # set to 1_000..10_000 if you see chatter

# -----------------------------
# Helpers
# -----------------------------
def to_bgra_array(image):
    """carla.Image -> BGRA uint8 numpy array"""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    return array.reshape((image.height, image.width, 4))

def carla_rgb_to_bgr(image):
    """carla.Image (RGBA) -> BGR uint8"""
    bgra = to_bgra_array(image)
    bgr = bgra[:, :, :3]
    return bgr

def dvs_events_to_image(events, w, h, t0, t1):
    """
    events: structured np array with fields ('x','y','t','pol')
    Build a 3-channel BGR image:
        positive polarity (pol=1) -> G channel
        negative polarity (pol=0) -> R channel
    """
    if events.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    # Select window
    m = (events['t'] >= t0) & (events['t'] < t1)
    e = events[m]
    
    img_r = np.zeros((h, w), dtype=np.float32)
    img_g = np.zeros((h, w), dtype=np.float32)

    # Clip to bounds (defensive)
    xs = np.clip(e['x'], 0, w - 1)
    ys = np.clip(e['y'], 0, h - 1)
    pol = e['pol']

    # Accumulate counts
    # pol=1 is positive (brightness increase) -> green
    # pol=0 is negative (brightness decrease) -> red
    pos = pol == 1
    neg = pol == 0
    
    if np.any(pos):
        np.add.at(img_g, (ys[pos], xs[pos]), 1.0)
    if np.any(neg):
        np.add.at(img_r, (ys[neg], xs[neg]), 1.0)

    # Simple normalization (per-channel to 0..255)
    def normalize(img):
        if img.max() <= 0:
            return np.zeros_like(img, dtype=np.uint8)
        # use sqrt to de-emphasize hot pixels
        img = np.sqrt(img / img.max())
        return (img * 255.0).clip(0, 255).astype(np.uint8)

    g = normalize(img_g)
    r = normalize(img_r)
    b = np.zeros_like(r)

    # BGR output
    out = np.stack([b, g, r], axis=-1)
    return out

# -----------------------------
# Main
# -----------------------------
def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    original_settings = world.get_settings()

    # Synchronous world
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = WORLD_DT
    settings.max_substep_delta_time = WORLD_DT
    settings.max_substeps = 1
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.global_percentage_speed_difference(-5.0)

    blueprint_library = world.get_blueprint_library()

    # Spawn ego vehicle
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    ego = world.spawn_actor(vehicle_bp, spawn_points[0])

    # RGB camera
    rgb_bp = blueprint_library.find("sensor.camera.rgb")
    rgb_bp.set_attribute("image_size_x", str(IMG_W))
    rgb_bp.set_attribute("image_size_y", str(IMG_H))
    rgb_bp.set_attribute("fov", str(FOV))
    rgb_bp.set_attribute("sensor_tick", f"{RGB_TICK:.3f}")

    # DVS camera
    dvs_bp = blueprint_library.find("sensor.camera.dvs")
    dvs_bp.set_attribute("image_size_x", str(IMG_W))
    dvs_bp.set_attribute("image_size_y", str(IMG_H))
    dvs_bp.set_attribute("fov", str(FOV))
    dvs_bp.set_attribute("sensor_tick", f"{DVS_TICK:.3f}")
    dvs_bp.set_attribute("positive_threshold", POS_TH)
    dvs_bp.set_attribute("negative_threshold", NEG_TH)
    dvs_bp.set_attribute("sigma_positive_threshold", "0.0")
    dvs_bp.set_attribute("sigma_negative_threshold", "0.0")
    dvs_bp.set_attribute("refractory_period_ns", REFRACTORY_NS)
    dvs_bp.set_attribute("use_log", "true")
    dvs_bp.set_attribute("log_eps", "0.001")

    # Mount both sensors
    cam_tf = carla.Transform(carla.Location(x=1.3, z=1.6), carla.Rotation(pitch=-5))
    rgb = world.spawn_actor(rgb_bp, cam_tf, attach_to=ego)
    dvs = world.spawn_actor(dvs_bp, cam_tf, attach_to=ego)

    # Buffers
    latest_rgb = {"img": None, "t": None}
    events_buffer = deque(maxlen=2000)  # (np_struct_array, slice_sim_ts)

    def on_rgb(image):
        latest_rgb["img"] = carla_rgb_to_bgr(image)
        latest_rgb["t"] = image.timestamp  # sim time of this RGB frame

    def on_dvs(dvs_array):
        # In CARLA 0.9.15, DVS events use a different format
        try:
            raw = np.frombuffer(dvs_array.raw_data, dtype=np.uint8)
            
            # Try different possible event sizes
            for event_size in [13, 14, 16, 12]:
                if len(raw) % event_size == 0:
                    break
            else:
                # No valid event size found
                return
            
            num_events = len(raw) // event_size
            if num_events == 0:
                return
            
            # For event_size=13, try alternative layout:
            # Maybe: x(2), y(2), t(8), pol(1) = 13 bytes
            # Let's try reading as structured array directly
            if event_size == 13:
                # Try struct: uint16, uint16, int64, int8
                dtype = np.dtype([
                    ('x', np.uint16),
                    ('y', np.uint16),
                    ('t', np.int64),
                    ('pol', np.int8)
                ])
                if dtype.itemsize == 13:
                    events_raw = np.frombuffer(dvs_array.raw_data, dtype=dtype)
                    
                    ev = np.zeros(len(events_raw), dtype=[
                        ('x', np.uint16),
                        ('y', np.uint16),
                        ('t', np.float64),
                        ('pol', np.int16)
                    ])
                    
                    ev['x'] = events_raw['x']
                    ev['y'] = events_raw['y']
                    # The timestamps in events are in nanoseconds (not microseconds!)
                    # Convert to seconds to match dvs_array.timestamp scale
                    ev['t'] = events_raw['t'] / 1e9
                    ev['pol'] = events_raw['pol'].astype(np.int16)
                    
                    events_buffer.append((ev, dvs_array.timestamp))
                    return
            
            # Fallback: manual parsing
            raw = raw.reshape((num_events, event_size))
            
            ev = np.zeros(num_events, dtype=[
                ('x', np.uint16),
                ('y', np.uint16),
                ('t', np.float64),
                ('pol', np.int16)
            ])
            
            # Extract x, y (first 2 bytes each, little-endian)
            ev['x'] = raw[:, 0] + (raw[:, 1] << 8)
            ev['y'] = raw[:, 2] + (raw[:, 3] << 8)
            
            # Extract timestamp - 8 bytes starting at offset 4 (microseconds)
            if event_size >= 12:
                t_bytes = raw[:, 4:12]
                t_us = np.zeros(num_events, dtype=np.int64)
                for i in range(8):
                    t_us += t_bytes[:, i].astype(np.int64) << (8 * i)
                # Convert to seconds and make absolute
                ev['t'] = t_us / 1e6 + dvs_array.timestamp
            
            # Extract polarity - at byte 12 (should be 0 or 1, or -1/+1)
            if event_size >= 13:
                pol_byte = raw[:, 12].astype(np.int16)
                # CARLA DVS: polarity is +1 for brightness increase, -1 for decrease
                # The byte might be 0/1 or already -1/+1
                ev['pol'] = np.where(pol_byte == 0, -1, 1)
            
            # Debug: print stats for first batch
            if len(events_buffer) == 0:
                print(f"First DVS batch: {num_events} events, event_size={event_size}")
                print(f"  X range: [{ev['x'].min()}, {ev['x'].max()}]")
                print(f"  Y range: [{ev['y'].min()}, {ev['y'].max()}]")
                print(f"  Pol values: {np.unique(ev['pol'])}")
                print(f"  Time range: [{ev['t'].min():.6f}, {ev['t'].max():.6f}]")
            
            events_buffer.append((ev, dvs_array.timestamp))
        except Exception as e:
            print(f"DVS parsing error: {e}")
            import traceback
            traceback.print_exc()
            return

    rgb.listen(on_rgb)
    dvs.listen(on_dvs)

    # Give the ego some motion (simple autopilot)
    ego.set_autopilot(True, tm.get_port())

    # Setup video writer
    os.makedirs("out", exist_ok=True)
    video_path = os.path.join("out", "rgb_dvs_side_by_side.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (IMG_W * 2, IMG_H))
    
    frames_written = 0
    last_frame_time = None
    
    # Tick loop
    start = world.get_snapshot().timestamp.elapsed_seconds
    try:
        print(f"Recording {DRIVE_SECONDS}s video at {VIDEO_FPS} FPS...")
        while True:
            world.tick()
            ts = world.get_snapshot().timestamp.elapsed_seconds
            
            # Check if we should save a frame (at RGB_TICK rate)
            rgb_img = latest_rgb["img"]
            t_rgb = latest_rgb["t"]
            
            if rgb_img is not None and t_rgb is not None:
                # Only process if this is a new frame
                if last_frame_time is None or t_rgb != last_frame_time:
                    last_frame_time = t_rgb
                    
                    # Get all events and create DVS image
                    if len(events_buffer) > 0:
                        evs = [ev for (ev, _) in events_buffer]
                        evs = np.concatenate(evs, axis=0) if len(evs) > 1 else evs[0]
                        
                        # Window
                        t0 = t_rgb - EVENT_WINDOW
                        t1 = t_rgb
                        dvs_img = dvs_events_to_image(evs, IMG_W, IMG_H, t0, t1)
                        
                        # Overlays
                        pos_cnt = int(np.sum((evs['t'] >= t0) & (evs['t'] < t1) & (evs['pol'] == 1)))
                        neg_cnt = int(np.sum((evs['t'] >= t0) & (evs['t'] < t1) & (evs['pol'] == 0)))
                        
                        rgb_annot = rgb_img.copy()
                        cv2.putText(rgb_annot, f"RGB @ {t_rgb:.2f}s", (10, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(dvs_img, f"DVS [{t0:.2f},{t1:.2f})  +:{pos_cnt}  -:{neg_cnt}", (10, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Side-by-side
                        combo = np.hstack([rgb_annot, dvs_img])
                        video_writer.write(combo)
                        frames_written += 1
            
            if ts - start >= DRIVE_SECONDS:
                break
                
        print(f"Wrote {frames_written} frames to {video_path}")
    finally:
        # Clean up
        video_writer.release()
        rgb.stop(); dvs.stop()
        ego.set_autopilot(False)
        rgb.destroy(); dvs.destroy(); ego.destroy()
        # Restore world settings
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)

if __name__ == "__main__":
    main()

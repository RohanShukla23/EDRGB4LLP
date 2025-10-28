"""
CARLA Speed Sweep - QUEUE-BASED CAPTURE 
=========================================================
We're using a queue to collect images synchronously.
This is the bulletproof approach that ALWAYS works.
"""

import sys
import os
import csv
import math
import pathlib
import json
import queue

EGG = r"C:\Users\jakep\Desktop\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg"
if EGG not in sys.path:
    sys.path.append(EGG)

import carla

# ═══════════════════════════════════════════════════════════════════
TOWN_NAME = "Town04"
OUT_BASE = pathlib.Path("_out")
SCENE_SECONDS = 12.0
TARGET_SPEEDS_KMH = [20, 40, 60]
FIXED_DT = 0.05
IMAGE_W, IMAGE_H = 1280, 720
FOV = 90
LOOKAHEAD_M = 8.0
STEER_GAIN = 1.6
THROTTLE_P = 0.35
MAX_STEER = 0.7
# ═══════════════════════════════════════════════════════════════════

def kmh_to_mps(v): return v / 3.6
def mps_to_kmh(v): return v * 3.6

def get_speed_mps(vehicle):
    v = vehicle.get_velocity()
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def pure_pursuit_steer(world, vehicle):
    amap = world.get_map()
    wp = amap.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if not wp: return 0.0
    ahead = wp.next(LOOKAHEAD_M)
    if not ahead: return 0.0
    target = ahead[0].transform.location
    tf = vehicle.get_transform()
    dx = target.x - tf.location.x
    dy = target.y - tf.location.y
    yaw = math.radians(tf.rotation.yaw)
    vx = math.cos(yaw)*dx + math.sin(yaw)*dy
    vy = -math.sin(yaw)*dx + math.cos(yaw)*dy
    steer = STEER_GAIN * math.atan2(vy, max(vx, 1e-3))
    return max(-MAX_STEER, min(MAX_STEER, steer))

def apply_speed_control(vehicle, target_kmh):
    target_mps = kmh_to_mps(target_kmh)
    current_mps = get_speed_mps(vehicle)
    error = target_mps - current_mps
    if error > 2.0: return 1.0, 0.0
    elif error < -2.0: return 0.0, min(abs(error)*0.2, 1.0)
    else: return max(0.0, min(1.0, THROTTLE_P*error)), 0.0

def main():
    OUT_BASE.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("CARLA Speed Sweep - QUEUE-BASED (THE FIX)")
    print("=" * 70)
    
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    print(f"✓ Connected to CARLA {client.get_server_version()}")
    
    # Load town
    current_map = world.get_map().name.split('/')[-1]
    if current_map != TOWN_NAME:
        print(f"Loading {TOWN_NAME}...")
        world = client.load_world(TOWN_NAME)
    else:
        print(f"✓ Already in {TOWN_NAME}")
    
    # Sync mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)
    print(f"✓ Synchronous mode (dt={FIXED_DT}s)")
    
    try:
        # Spawn vehicle
        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        veh_bp = bp_lib.find("vehicle.tesla.model3")
        vehicle = world.spawn_actor(veh_bp, spawn_points[0])
        print(f"✓ Vehicle spawned (id={vehicle.id})")
        vehicle.set_autopilot(False)
        
        # Warm up
        for _ in range(10):
            world.tick()
        print("✓ Warmed up\n")
        
        total_ticks = int(SCENE_SECONDS / FIXED_DT)
        
        for run_idx, target_speed in enumerate(TARGET_SPEEDS_KMH):
            print("=" * 70)
            print(f"RUN {run_idx+1}/{len(TARGET_SPEEDS_KMH)}: {target_speed} km/h")
            print("=" * 70)
            
            run_dir = OUT_BASE / f"{target_speed}kmh"
            run_dir.mkdir(exist_ok=True)
            
            # Reset vehicle
            vehicle.set_transform(spawn_points[0])
            vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            for _ in range(5):
                world.tick()
            
            # ============================================================
            # SPAWN CAMERA WITH QUEUE (NO CALLBACK)
            # ============================================================
            cam_bp = bp_lib.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(IMAGE_W))
            cam_bp.set_attribute("image_size_y", str(IMAGE_H))
            cam_bp.set_attribute("fov", str(FOV))
            # DON'T set sensor_tick - let CARLA handle it
            
            cam_tf = carla.Transform(carla.Location(x=1.5, z=2.2))
            camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
            
            # Create image queue
            image_queue = queue.Queue()
            camera.listen(image_queue.put)
            
            print(f"✓ Camera ready with queue")
            
            # Warm up camera
            for _ in range(5):
                world.tick()
                try:
                    image_queue.get(timeout=0.1)
                except queue.Empty:
                    pass
            
            print(f"🎬 Recording {total_ticks} frames...")
            
            # Open metadata
            metadata_path = run_dir / "metadata.csv"
            with open(metadata_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame", "sim_time_s", "speed_mps", "speed_kmh", "target_kmh", "error_kmh"])
                
                speed_errors = []
                sim_time = 0.0
                
                # MAIN LOOP - QUEUE-BASED
                for tick_idx in range(total_ticks):
                    # Control
                    steer = pure_pursuit_steer(world, vehicle)
                    throttle, brake = apply_speed_control(vehicle, target_speed)
                    vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
                    
                    # Step simulation
                    world.tick()
                    sim_time += FIXED_DT
                    
                    # Get image from queue (BLOCKING - will wait for it)
                    try:
                        image = image_queue.get(timeout=2.0)
                        
                        # Save image
                        image_path = run_dir / f"{tick_idx:06d}.png"
                        image.save_to_disk(str(image_path))
                        
                    except queue.Empty:
                        print(f"  ⚠️  WARNING: No image at tick {tick_idx}")
                    
                    # Measure speed
                    speed_mps = get_speed_mps(vehicle)
                    speed_kmh = mps_to_kmh(speed_mps)
                    error = target_speed - speed_kmh
                    speed_errors.append(abs(error))
                    
                    # Log
                    writer.writerow([tick_idx, f"{sim_time:.3f}", f"{speed_mps:.3f}", 
                                   f"{speed_kmh:.2f}", target_speed, f"{error:.2f}"])
                    
                    # Progress
                    if (tick_idx + 1) % 40 == 0:
                        print(f"  [{tick_idx+1:3d}/{total_ticks}] t={sim_time:.1f}s  speed={speed_kmh:.1f} km/h")
            
            # Cleanup camera
            camera.stop()
            camera.destroy()
            world.tick()
            
            # Count frames
            frames_saved = len(list(run_dir.glob("*.png")))
            mean_error = sum(speed_errors) / len(speed_errors)
            max_error = max(speed_errors)
            
            # Summary
            summary = {
                "target_speed_kmh": target_speed,
                "frames_expected": total_ticks,
                "frames_saved": frames_saved,
                "match": frames_saved == total_ticks,
                "mean_speed_error_kmh": round(mean_error, 2),
                "max_speed_error_kmh": round(max_error, 2)
            }
            
            with open(run_dir / "run_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            
            print(f"\n✓ COMPLETE: {frames_saved}/{total_ticks} frames {'✓' if summary['match'] else '✗'}")
            print(f"  Speed error: mean={mean_error:.2f}, max={max_error:.2f} km/h\n")
        
        print("=" * 70)
        print("ALL RUNS COMPLETE")
        print("=" * 70)
        
        # Cleanup
        vehicle.destroy()
        
    finally:
        world.apply_settings(original_settings)
        print("✓ Cleanup done")

if __name__ == "__main__":
    main()
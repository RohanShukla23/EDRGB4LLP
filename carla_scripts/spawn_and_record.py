# --- Make CARLA egg importable on Windows ---
import sys
egg = r"C:\Users\jakep\Desktop\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.15-py3.7-win-amd64.egg"
if egg not in sys.path:
    sys.path.append(egg)
# --------------------------------------------

import os
import time
import carla

OUT_DIR = r"_out"
os.makedirs(OUT_DIR, exist_ok=True)

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
settings = world.get_settings()

# Enable synchronous mode for determinism (important for research)
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05  # 20 FPS sim tick; adjust as needed
world.apply_settings(settings)

tm = client.get_trafficmanager()
tm.set_synchronous_mode(True)

blueprints = world.get_blueprint_library()

# Spawn ego vehicle
vehicle_bp = blueprints.find("vehicle.tesla.model3")
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
if vehicle is None:
    raise RuntimeError("Vehicle spawn failed; try a different spawn index.")

vehicle.set_autopilot(False)


# RGB camera
cam_bp = blueprints.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1280")
cam_bp.set_attribute("image_size_y", "720")
cam_bp.set_attribute("fov", "90")

cam_tf = carla.Transform(carla.Location(x=1.5, z=2.2))  # hood mount-ish
camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)

# Register callback
camera.listen(lambda img: img.save_to_disk(os.path.join(OUT_DIR, f"{img.frame:06d}.png")))

# Step the world for N ticks
try:
    print("Capturing ~10 seconds of frames...")
    ticks = int(10 / settings.fixed_delta_seconds)
    for _ in range(ticks):
        world.tick()     # advance one deterministic step
        # (Optional) sleep isn’t needed in sync mode, but small delay is fine:
        # time.sleep(settings.fixed_delta_seconds)
finally:
    # Cleanup in reverse order
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    # Restore async mode so editor isn’t stuck
    settings.synchronous_mode = False
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)
    print(f"Done. Frames saved to: {OUT_DIR}\\")

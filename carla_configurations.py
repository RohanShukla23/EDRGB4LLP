import carla
import random
import time
import numpy as np
import pandas as pd
import cv2
from collections import deque
import psutil
import os

class AVConfiguration:
    """Base class for AV sensor configurations"""
    def __init__(self, world, vehicle, config_name):
        self.world = world
        self.vehicle = vehicle
        self.config_name = config_name
        self.sensors = []
        self.data_queue = deque(maxlen=100)
        # Dictionary to store performance metrics for analysis
        self.metrics = {
            'frame': [],
            'timestamp': [],
            'latency_ms': [],
            'cpu_percent': [],
            'memory_mb': [],
            'detections': []
        }
        
    def cleanup(self):
        """Destroy all sensors attached to this configuration"""
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.destroy()

class DefaultAVConfig(AVConfiguration):
    """Default AV with only RGB camera"""
    def __init__(self, world, vehicle):
        super().__init__(world, vehicle, "Default_RGB_Only")
        self.setup_sensors()
        
    def setup_sensors(self):
        """Configure and spawn RGB camera sensor"""
        bp_lib = self.world.get_blueprint_library()
        camera_bp = bp_lib.find('sensor.camera.rgb')
        # Set camera resolution and field of view
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        # Set sensor tick to 20 Hz (0.05s between captures)
        camera_bp.set_attribute('sensor_tick', '0.05')
        
        # Mount camera on top of vehicle
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        camera.listen(lambda data: self.process_rgb(data))
        self.sensors.append(camera)
        
    def process_rgb(self, image):
        """Process RGB camera data and detect hazards"""
        start_time = time.time()
        
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Drop alpha channel
        
        # Run hazard detection on the image
        detections = self.detect_hazards_rgb(array)
        
        # Calculate processing latency
        latency = (time.time() - start_time) * 1000
        
        # Record metrics for this frame
        self.metrics['frame'].append(image.frame)
        self.metrics['timestamp'].append(image.timestamp)
        self.metrics['latency_ms'].append(latency)
        self.metrics['cpu_percent'].append(psutil.cpu_percent())
        self.metrics['memory_mb'].append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
        self.metrics['detections'].append(len(detections))
        
    def detect_hazards_rgb(self, image):
        """
        Detect pedestrians from RGB image using color thresholding.
        In production, this would use a trained neural network.
        """
        detections = []
        
        # Use HSV color space to detect skin tones (proxy for pedestrians)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Filter out small detections (noise)
            if cv2.contourArea(contour) > 500:
                detections.append({'type': 'pedestrian', 'area': cv2.contourArea(contour)})
                
        return detections

class FusionAVConfig(AVConfiguration):
    """Fusion AV with RGB + DVS Event Camera"""
    def __init__(self, world, vehicle):
        super().__init__(world, vehicle, "Fusion_RGB_DVS")
        self.rgb_data = None
        self.rgb_timestamp = None
        self.event_buffer = []
        self.last_fusion_time = time.time()
        self.fusion_interval = 0.05  # Match RGB sensor tick rate
        self.setup_sensors()
        
    def setup_sensors(self):
        """Configure and spawn both RGB and DVS event cameras"""
        bp_lib = self.world.get_blueprint_library()
        
        # RGB Camera setup (same as default config)
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        camera_bp.set_attribute('sensor_tick', '0.05')
        
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        camera.listen(lambda data: self.process_rgb(data))
        self.sensors.append(camera)
        
        # DVS Event Camera setup
        dvs_bp = bp_lib.find('sensor.camera.dvs')
        dvs_bp.set_attribute('image_size_x', '800')
        dvs_bp.set_attribute('image_size_y', '600')
        dvs_bp.set_attribute('fov', '90')
        # Threshold for triggering positive/negative events
        dvs_bp.set_attribute('positive_threshold', '0.3')
        dvs_bp.set_attribute('negative_threshold', '0.3')
        dvs_bp.set_attribute('sigma_positive_threshold', '0')
        dvs_bp.set_attribute('sigma_negative_threshold', '0')
        dvs_bp.set_attribute('use_log', 'true')
        # Continuous stream (no tick limit) - events processed asynchronously
        dvs_bp.set_attribute('sensor_tick', '0.0')
        
        dvs = self.world.spawn_actor(dvs_bp, camera_transform, attach_to=self.vehicle)
        dvs.listen(lambda data: self.process_dvs(data))
        self.sensors.append(dvs)
        
    def process_rgb(self, image):
        """Store RGB frame for later fusion with event data"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.rgb_data = array[:, :, :3]
        self.rgb_timestamp = image.timestamp
        
    def process_dvs(self, data):
        """
        Process DVS event stream. Events are buffered and periodically
        fused with RGB data at the fusion interval rate.
        """
        # Parse DVS events from raw data
        events = np.frombuffer(data.raw_data, dtype=np.dtype([
            ('x', np.uint16),
            ('y', np.uint16),
            ('t', np.int64),
            ('pol', np.bool_)
        ]))
        
        # Add events to buffer for accumulation
        self.event_buffer.extend(events)
        
        # Check if it's time to perform fusion
        current_time = time.time()
        if current_time - self.last_fusion_time >= self.fusion_interval:
            self.perform_fusion(data.frame, data.timestamp)
            self.last_fusion_time = current_time
            # Clear buffer after fusion to start fresh
            self.event_buffer = []
    
    def perform_fusion(self, frame, timestamp):
        """
        Execute sensor fusion between RGB and DVS data.
        This is where the actual fusion magic happens.
        """
        # Skip if we don't have both data sources ready
        if self.rgb_data is None or len(self.event_buffer) == 0:
            return
        
        start_time = time.time()
        
        # Convert event list to numpy array for processing
        events = np.array(self.event_buffer)
        # Run fusion-based detection on combined data
        detections = self.detect_hazards_fusion(self.rgb_data, events)
        
        # Calculate fusion processing latency
        latency = (time.time() - start_time) * 1000
        
        # Record metrics for this fusion cycle
        self.metrics['frame'].append(frame)
        self.metrics['timestamp'].append(timestamp)
        self.metrics['latency_ms'].append(latency)
        self.metrics['cpu_percent'].append(psutil.cpu_percent())
        self.metrics['memory_mb'].append(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024)
        self.metrics['detections'].append(len(detections))
        
    def detect_hazards_fusion(self, rgb_image, events):
        """
        Combined detection using both RGB and event data.
        This leverages RGB for appearance-based detection and
        DVS events for motion-based detection.
        """
        detections = []
        
        # First, run traditional RGB-based detection
        rgb_detections = self.detect_from_rgb(rgb_image)
        
        # Then process event data if available
        if len(events) > 0:
            # Create event frame by accumulating events
            event_frame = np.zeros((600, 800), dtype=np.uint8)
            for event in events:
                if 0 <= event['y'] < 600 and 0 <= event['x'] < 800:
                    # Positive events = bright pixels, negative = darker pixels
                    event_frame[event['y'], event['x']] = 255 if event['pol'] else 128
            
            # Detect moving objects from event frame
            event_detections = self.detect_from_events(event_frame)
            
            # Merge detections from both modalities
            detections = self.merge_detections(rgb_detections, event_detections)
        else:
            # Fall back to RGB-only if no events
            detections = rgb_detections
            
        return detections
    
    def detect_from_rgb(self, image):
        """Detect pedestrians from RGB using color thresholding"""
        detections = []
        
        # Use skin tone detection as proxy for pedestrians
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                # Calculate centroid for spatial matching later
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    detections.append({
                        'type': 'rgb_pedestrian',
                        'x': cx,
                        'y': cy,
                        'area': cv2.contourArea(contour)
                    })
        
        return detections
    
    def detect_from_events(self, event_frame):
        """Detect moving objects from accumulated DVS events"""
        detections = []
        
        # Threshold the event frame to find significant activity
        _, thresh = cv2.threshold(event_frame, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Lower threshold than RGB since events indicate motion
            if cv2.contourArea(contour) > 200:
                # Calculate centroid
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    detections.append({
                        'type': 'event_motion',
                        'x': cx,
                        'y': cy,
                        'area': cv2.contourArea(contour)
                    })
        
        return detections
    
    def merge_detections(self, rgb_detections, event_detections):
        """
        Merge RGB and event detections, avoiding duplicates.
        If an event detection is near an RGB detection, it's likely the same object.
        """
        merged = list(rgb_detections)
        proximity_threshold = 50  # pixels
        
        for event_det in event_detections:
            # Check if this event detection is near any RGB detection
            is_near_rgb = False
            for rgb_det in rgb_detections:
                # Calculate Euclidean distance between centroids
                dist = np.sqrt((event_det['x'] - rgb_det['x'])**2 + 
                             (event_det['y'] - rgb_det['y'])**2)
                if dist < proximity_threshold:
                    is_near_rgb = True
                    break
            
            # Only add event detection if it's a new, unique object
            if not is_near_rgb:
                merged.append(event_det)
        
        return merged

def setup_urban_environment(client, town_name='Town03'):
    """
    Setup realistic urban environment with traffic and pedestrians.
    Uses synchronous mode for deterministic, reproducible experiments.
    """
    world = client.load_world(town_name)
    
    # Configure realistic weather conditions
    weather = carla.WeatherParameters(
        cloudiness=30.0,
        precipitation=0.0,
        sun_altitude_angle=70.0,
        fog_density=10.0
    )
    world.set_weather(weather)
    
    # Enable synchronous mode for deterministic simulation
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    
    # Spawn traffic vehicles
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bps = world.get_blueprint_library().filter('vehicle.*')
    
    traffic_vehicles = []
    for i in range(30):
        vehicle_bp = random.choice(vehicle_bps)
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True)
            traffic_vehicles.append(vehicle)
    
    # Spawn pedestrians
    walker_bps = world.get_blueprint_library().filter('walker.pedestrian.*')
    walker_spawn_points = []
    for i in range(20):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_point.location = loc
            walker_bp = random.choice(walker_bps)
            walker = world.try_spawn_actor(walker_bp, spawn_point)
            if walker:
                walker_spawn_points.append(walker)
    
    return world, traffic_vehicles, walker_spawn_points

def run_experiment_on_world(world, spawn_point, config_class, duration_seconds=60, recorded_controls=None):
    """
    Run experiment using an existing world and fixed spawn point.
    
    If recorded_controls is None, this records the vehicle's control inputs.
    If recorded_controls is provided, this replays those exact controls.
    This ensures both configurations follow identical trajectories.
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {config_class.__name__}")
    print(f"{'='*60}\n")

    # Spawn ego vehicle at predetermined location
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    # Only use autopilot for recording run
    if recorded_controls is None:
        ego_vehicle.set_autopilot(True)

    # Attach sensor configuration to ego vehicle
    config = config_class(world, ego_vehicle)

    start_time = time.time()
    frame_count = 0
    controls_record = [] if recorded_controls is None else None

    try:
        # Run simulation for specified duration
        while time.time() - start_time < duration_seconds:
            world.tick()  # Advance simulation by one step
            
            # Record or replay vehicle controls
            if recorded_controls is None:
                # Recording mode: capture control inputs from autopilot
                control = ego_vehicle.get_control()
                controls_record.append({
                    'throttle': control.throttle,
                    'steer': control.steer,
                    'brake': control.brake,
                    'hand_brake': control.hand_brake,
                    'reverse': control.reverse,
                    'manual_gear_shift': control.manual_gear_shift,
                    'gear': control.gear
                })
            else:
                # Replay mode: apply recorded control inputs
                if frame_count < len(recorded_controls):
                    ctrl = recorded_controls[frame_count]
                    control = carla.VehicleControl(
                        throttle=ctrl['throttle'],
                        steer=ctrl['steer'],
                        brake=ctrl['brake'],
                        hand_brake=ctrl['hand_brake'],
                        reverse=ctrl['reverse'],
                        manual_gear_shift=ctrl['manual_gear_shift'],
                        gear=ctrl['gear']
                    )
                    ego_vehicle.apply_control(control)
            
            frame_count += 1

            # Print progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {elapsed:.1f}s / {duration_seconds}s ({frame_count} frames)")

    finally:
        # Cleanup ego vehicle and sensors (leave traffic for next config)
        config.cleanup()
        if ego_vehicle.is_alive:
            ego_vehicle.destroy()

    return config.metrics, controls_record

def save_metrics_to_csv(metrics_dict, output_dir='results'):
    """
    Save performance metrics to CSV files for analysis.
    Creates both detailed per-frame metrics and summary statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for config_name, metrics in metrics_dict.items():
        df = pd.DataFrame(metrics)
        
        # Calculate summary statistics
        summary = {
            'Configuration': config_name,
            'Avg_Latency_ms': df['latency_ms'].mean(),
            'Std_Latency_ms': df['latency_ms'].std(),
            'Max_Latency_ms': df['latency_ms'].max(),
            'Avg_CPU_percent': df['cpu_percent'].mean(),
            'Avg_Memory_MB': df['memory_mb'].mean(),
            'Total_Detections': df['detections'].sum(),
            'Avg_Detections_per_frame': df['detections'].mean()
        }
        
        # Save detailed frame-by-frame metrics
        filename = f"{output_dir}/{config_name}_detailed.csv"
        df.to_csv(filename, index=False)
        print(f"Saved detailed metrics to: {filename}")
        
        # Save summary statistics
        summary_df = pd.DataFrame([summary])
        summary_filename = f"{output_dir}/{config_name}_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        print(f"Saved summary to: {summary_filename}")
    
    # Create side-by-side comparison table
    comparison_data = []
    for config_name, metrics in metrics_dict.items():
        df = pd.DataFrame(metrics)
        comparison_data.append({
            'Configuration': config_name,
            'Avg_Latency_ms': df['latency_ms'].mean(),
            'Avg_CPU_%': df['cpu_percent'].mean(),
            'Avg_Memory_MB': df['memory_mb'].mean(),
            'Total_Detections': df['detections'].sum()
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_filename = f"{output_dir}/comparison.csv"
    comparison_df.to_csv(comparison_filename, index=False)
    print(f"\nSaved comparison table to: {comparison_filename}")
    print("\nComparison Results:")
    print(comparison_df.to_string(index=False))

def main():
    # Connect to CARLA simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Seed for reproducibility across runs
    random.seed(42)

    print("\n" + "="*60)
    print("STAGE 1: Recording baseline trajectory")
    print("="*60)
    
    # Setup environment and record the baseline trajectory
    world, traffic_vehicles, pedestrians = setup_urban_environment(client)
    spawn_points = world.get_map().get_spawn_points()
    ego_spawn_point = random.choice(spawn_points)

    # First run: Record the trajectory using DefaultAVConfig
    metrics_default, recorded_controls = run_experiment_on_world(
        world, ego_spawn_point, DefaultAVConfig, duration_seconds=60, recorded_controls=None
    )
    
    print("\n" + "="*60)
    print(f"STAGE 2: Resetting world and replaying trajectory")
    print("="*60)
    
    # Clean up the first run
    for vehicle in traffic_vehicles:
        if vehicle.is_alive:
            vehicle.destroy()
    for walker in pedestrians:
        if walker.is_alive:
            walker.destroy()
    
    # Reset the environment to exact same initial state
    world, traffic_vehicles, pedestrians = setup_urban_environment(client)
    
    # Second run: Replay exact same trajectory with FusionAVConfig
    metrics_fusion, _ = run_experiment_on_world(
        world, ego_spawn_point, FusionAVConfig, duration_seconds=60, recorded_controls=recorded_controls
    )

    # Final cleanup
    for vehicle in traffic_vehicles:
        if vehicle.is_alive:
            vehicle.destroy()
    for walker in pedestrians:
        if walker.is_alive:
            walker.destroy()

    # Save all metrics to CSV files
    metrics_dict = {
        'Default_RGB_Only': metrics_default,
        'Fusion_RGB_DVS': metrics_fusion
    }
    save_metrics_to_csv(metrics_dict)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as e:
        print(f'Error: {e}')
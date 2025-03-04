import carla
import random
import pygame
import torch
import numpy as np
import gym
from gym import spaces
import os
import sys
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Expand the CARLA_ROOT environment variable correctly:
carla_root = os.environ.get("CARLA_ROOT")
if carla_root is None:
    raise EnvironmentError("CARLA_ROOT environment variable is not set.")
sys.path.append(os.path.join(carla_root, "PythonAPI", "carla"))

# Import the default VehiclePIDController from CARLA
from agents.navigation.controller import VehiclePIDController
from agents.navigation.global_route_planner import GlobalRoutePlanner

# MAI imports
from vector_BEV_observer import Vector_BEV_observer
from dipp_predictor_py.dipp_carla import Predictor
from carla_env_render import MatplotlibAnimationRenderer


# Global parameters (default values)
N_VEHICLES = 30
SCENE_DURATION = 1 * 60  # seconds
SLOWDOWN_PERCENTAGE = 10
EGO_AUTOPILOT = False
FOLLOW_POINT_DIST = 8   # meters (used in custom_sample_action)
REQ_TIME = 1            # seconds
FREQUENCY = 0.1         # simulation tick time
RENDER_CAMERA = False
USE_CUSTOM_MAP = False
NUM_ACTIONS = 3
N_ACTION_PER_MANEUVER = 5
SHOW_ROUTE = True
TRAIN = True
TEST = False
LOAD_MODEL = False
display_width = 1080
display_height = 720


class CarlaGymEnv(gym.Env):
    """
    A Gym environment wrapping a CARLA simulation. Observations are a dictionary 
    with keys "ego", "neighbors", and "map". The action is a 2D offset (in the 
    ego coordinate system) that is rotated using the current ego yaw.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, render_enabled=False):
        """
        Initialize the environment.
        
        Parameters:
            render_enabled (bool): If True, the camera sensor is spawned and
                                   its output rendered via Pygame.
        """
        super(CarlaGymEnv, self).__init__()
        self.render_enabled = render_enabled

        # Simulation parameters
        self.SCENE_DURATION = SCENE_DURATION
        self.SLOWDOWN_PERCENTAGE = SLOWDOWN_PERCENTAGE
        self.EGO_AUTOPILOT = EGO_AUTOPILOT
        self.FOLLOW_POINT_DIST = FOLLOW_POINT_DIST
        self.REQ_TIME = REQ_TIME
        self.FREQUENCY = FREQUENCY
        self.USE_CUSTOM_MAP = USE_CUSTOM_MAP
        self.SHOW_ROUTE = SHOW_ROUTE
        self.sim_time = 0.0
        self.map_path = '/home/ratul/Downloads/Tegel_map_for_Decision_1302.xodr'
        
        # Pygame setup for camera display (only if rendering enabled)
        if self.render_enabled:
            pygame.init()
            self.screen = pygame.display.set_mode((display_width, display_height))
            pygame.display.set_caption("Carla Gym Environment")
        self.display_width = display_width
        self.display_height = display_height

        # Connect to CARLA server and get world
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        if USE_CUSTOM_MAP:
            # Load the custom OpenDRIVE (.xodr) map
            with open(self.map_path, 'r') as f:
                opendrive_data = f.read()

            # Generate the world using the OpenDRIVE map
            self.world = self.client.generate_opendrive_world(
                # opendrive_data,  # commented out, fix if needed
                carla.OpendriveGenerationParameters(
                    wall_height=0.0, 
                    additional_width=0.0, 
                    smooth_junctions=True,
                    enable_mesh_visibility=True
                ), 
                reset_settings=True
            )
            print("Successfully loaded custom map:", self.map_path)
        else:
            # load default world
            self.world = self.client.get_world()

        self.blueprint_library = self.world.get_blueprint_library()
        
        # Set synchronous mode and fixed delta time
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.FREQUENCY
        self.world.apply_settings(settings)

        # Traffic manager
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.global_percentage_speed_difference(self.SLOWDOWN_PERCENTAGE)

        # Initialize BEV observer (your original code uses FUTURE_LEN=1)
        self.bev_info = Vector_BEV_observer(FUTURE_LEN=1)
        
        # For cleanup, track spawned actors
        self.actor_list = []

        # Define the action space early on:
        # self.action_space = spaces.Box(low=5, high=10, shape=(2,), dtype=np.float32)
        self.NUM_MANEUVERS = NUM_ACTIONS  # Possible actions: 0, 1, 2, 3, 4
        self.N_ACTION_PER_MANEUVER = N_ACTION_PER_MANEUVER
        self.action_space = spaces.MultiDiscrete([self.NUM_MANEUVERS, self.N_ACTION_PER_MANEUVER])
        # Define the possible values for each dimension
        self.index_map = {1: 1, 2: 5, 3: 10, 4: 15}
        # --------------------------------------------------------
        # Hard-code your observation space to match the shapes you expect.
        # Example shapes (leading dimension 1 for ego, 5 for neighbors, etc.).
        # Adjust these as needed to match your real data.
        # --------------------------------------------------------
        self.observation_space = spaces.Dict({
            "ego": spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.bev_info.HISTORY, 24), dtype=np.float32),
            "neighbors": spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.bev_info.MAX_NEIGHBORS, self.bev_info.HISTORY, 24), dtype=np.float32),
            "map": spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.bev_info.MAX_LANES, self.bev_info.MAX_LANE_LEN, 46), dtype=np.float32),
            "global_route": spaces.Box(low=-np.inf, high=np.inf, shape=(self.bev_info.MAX_LANE_LEN, 3), dtype=np.float32)
        })

        # Initialize collision flag
        self.collision_detected = False

        # Call reset to start the simulation.
        self.reset()

    def seed(self, seed=None):
        """
        Set the random seed for Python, NumPy, and Torch for reproducibility.
        Returns a list with the seed used.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]
    
    def process_image(self, image, screen):
        """
        Convert the CARLA image to a numpy array, then to a pygame surface,
        and display it.
        """
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]  # Convert from BGRA to RGB
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
    
    def ego_to_global(self, action, ego_position, ego_yaw):
        """
        Convert a 2D action (offset in ego frame) into a global coordinate.
        """
        cos_theta = np.cos(ego_yaw)
        sin_theta = np.sin(ego_yaw)
        R = np.array([[cos_theta, -sin_theta],
                      [sin_theta, cos_theta]])
        global_point = (R @ action.reshape(2, 1)).reshape(2,) + ego_position
        return global_point

    def _on_collision(self, event):
        """
        Callback for collision events. Sets the collision flag.
        """
        self.collision_detected = True

    def _cleanup(self):
        """
        Destroy all actors (vehicles, sensors, etc.) that were spawned.
        """
        print("Destroying actors...")
        for actor in self.actor_list:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self.actor_list = []

    def reset(self):
        """
        Reset the simulation: clean up previous actors, spawn the ego vehicle,
        attach sensors (including collision sensor and, if enabled, camera), spawn 
        other vehicles, and reset time.
        Returns an initial observation.
        """
        self._cleanup()
        self.sim_time = 0.0
        self.collision_detected = False

        # Initialize BEV observer (your original code uses FUTURE_LEN=1)
        self.bev_info = Vector_BEV_observer(FUTURE_LEN=1)

        # Spawn Ego Vehicle
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute("role_name", "hero")
        self.ego_vehicle = None
        while self.ego_vehicle is None:
            spawn_points = self.world.get_map().get_spawn_points()
            self.spawn_points = spawn_points
            self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        self.actor_list.append(self.ego_vehicle)
        if not self.EGO_AUTOPILOT:
            # Initialize the PID controller for the ego vehicle
            args_lateral = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.2, 'dt': 1.0 / 20.0}
            args_longitudinal = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.2, 'dt': 1.0 / 20.0}
            self.pid_controller = VehiclePIDController(self.ego_vehicle, 
                                                       args_lateral=args_lateral, 
                                                       args_longitudinal=args_longitudinal)
        else:
            self.ego_vehicle.set_autopilot(self.EGO_AUTOPILOT, self.tm.get_port())

        # Attach Camera to Ego Vehicle if rendering is enabled
        if self.render_enabled:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.display_width))
            camera_bp.set_attribute('image_size_y', str(self.display_height))
            camera_bp.set_attribute('fov', '90')
            camera_transform = carla.Transform(carla.Location(x=0, z=35.0), carla.Rotation(pitch=-90))
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
            self.actor_list.append(self.camera)
            self.camera.listen(lambda image: self.process_image(image, self.screen))

        # Attach Collision Sensor to Ego Vehicle
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.collision_sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.ego_vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        # Spawn other vehicles on autopilot
        self.vehicles = []
        for _ in range(N_VEHICLES):
            vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
            spawn_point = random.choice(spawn_points)
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                vehicle.set_autopilot(True, self.tm.get_port())
                self.tm.vehicle_percentage_speed_difference(vehicle, self.SLOWDOWN_PERCENTAGE)
                self.vehicles.append(vehicle)
                self.actor_list.append(vehicle)

        # Tick the world once to initialize everything
        self.world.tick()

        # Get initial observation from the BEV observer
        if self.bev_info.client_init(world=self.world):
            ego_hist, neighbor_hist, map_hist, crosswalk_hist, _ = self.bev_info.update(re_reference=False)
            ego_hist, neighbor_hist, map_hist, ground_truth, crosswalk_hist = \
                self.bev_info.create_buffer_and_transform_frame(np.array(ego_hist),
                                                                  np.array(neighbor_hist), 
                                                                  np.array(map_hist), 
                                                                  np.array(crosswalk_hist))
            # -----------
            # Convert your data to the shapes (1, 20, 24), (5, 20, 24), (20, 50, 46)
            # or whatever shapes you declared in observation_space
            # -----------
            ego_obs = self.bev_info.carla_to_MAI_coordinates(data=ego_hist, is_map=False)
            neighbors_obs = self.bev_info.carla_to_MAI_coordinates(data=neighbor_hist, is_map=False)
            map_obs = self.bev_info.carla_to_MAI_coordinates(data=map_hist, is_map=True)
            if len(ego_obs) > 1:
                ego_obs = ego_obs[-2:-1]
                neighbors_obs = neighbors_obs[-2:-1]
                map_obs = map_obs[-2:-1]

        else:
            ego_obs, neighbors_obs, map_obs = None, None, None

        self.global_route = None
        self.global_route_ego_frame = torch.zeros(size=(self.bev_info.MAX_LANE_LEN, 3))
        observation = {"ego": ego_obs, 
                       "neighbors": neighbors_obs, 
                       "map": map_obs, 
                       "global_route": self.global_route_ego_frame}  # New key with the global route.
        return observation

    def _generate_global_route(self):
        """
        Generate the global route using the GlobalRoutePlanner and convert it into the ego frame.
        Returns:
            A numpy array of shape (N, 3) with columns [x, y, relative_yaw] in the ego coordinate frame.
        """
        # Get the ego vehicle's current location.
        start_location = self.ego_vehicle.get_transform().location

        # Choose a destination from spawn_points that is at least 100m away.
        destination = random.choice(self.spawn_points)
        while destination.location.distance(start_location) < 100:
            destination = random.choice(self.spawn_points)

        # Set up the global route planner.
        grp = GlobalRoutePlanner(self.world.get_map(), 2.0)
        # trace_route returns a list of tuples (waypoint, road_option).
        route = grp.trace_route(start_location, destination.location)

        # Convert the global route into an array with columns [x, y, yaw].
        global_route_list = [
            [wp.transform.location.x, wp.transform.location.y, wp.transform.rotation.yaw]
            for wp, _ in route
        ]
        self.global_route = np.array(global_route_list)
        # Optionally store the original start and destination too.
        self.global_route_start = start_location
        self.global_route_destination = destination

        return self.global_route

    def _transform_to_ego_frame(self):
        # Now, convert the global route into the ego frame.
        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_rotation = ego_transform.rotation
        ego_x, ego_y = ego_location.x, ego_location.y
        ego_yaw_rad = np.deg2rad(ego_rotation.yaw)
        cos_yaw = np.cos(ego_yaw_rad)
        sin_yaw = np.sin(ego_yaw_rad)

        global_route_ego_frame = []
        for global_point in self.global_route:
            global_x, global_y, global_yaw = global_point
            # Translate: shift coordinates by subtracting the ego's position.
            dx = global_x - ego_x
            dy = global_y - ego_y
            # Rotate: apply the inverse rotation so that ego heading aligns with the x-axis.
            x_ego = dx * cos_yaw + dy * sin_yaw
            y_ego = -dx * sin_yaw + dy * cos_yaw
            # Relative yaw is the difference between the waypoint's yaw and the ego yaw.
            relative_yaw = global_yaw - ego_rotation.yaw
            global_route_ego_frame.append([x_ego, y_ego, relative_yaw])

        global_route_ego_frame = np.array(global_route_ego_frame)
        global_route_ego_frame = global_route_ego_frame[global_route_ego_frame[:, 0] >= 0] # keep only the positive route
        global_route_ego_frame_no_padding = global_route_ego_frame[:self.bev_info.MAX_LANE_LEN].copy()
        # make the route fixed size.
        if len(global_route_ego_frame) > self.bev_info.MAX_LANE_LEN:
            global_route_ego_frame = global_route_ego_frame[:self.bev_info.MAX_LANE_LEN]
        elif len(global_route_ego_frame) < self.bev_info.MAX_LANE_LEN:
            padd_len = self.bev_info.MAX_LANE_LEN - len(global_route_ego_frame)
            global_route_ego_frame = np.pad(global_route_ego_frame, ((0, padd_len), (0, 0)))
        return global_route_ego_frame, global_route_ego_frame_no_padding

    def step(self, action):
        # Compute the global route only once.
        if self.global_route is None and self.ego_vehicle.get_transform().location.x != 0.0:
            self.global_route = self._generate_global_route()
            if self.SHOW_ROUTE:
                for point in self.global_route:
                    point = carla.Location(x=float(point[0]), y=float(point[1]))
                    self.world.debug.draw_point(
                        point, 
                        size=0.1, 
                        color=carla.Color(255, 255, 0), 
                        life_time=self.SCENE_DURATION
                    )
        if self.global_route is not None:
            # transform global route to ego frame
            self.global_route_ego_frame, self.global_route_ego_frame_no_padding = self._transform_to_ego_frame()

        # Get current ego transform information
        ego_transform = self.ego_vehicle.get_transform()
        current_location = ego_transform.location
        ego_position_global = np.array([current_location.x, current_location.y])
        ego_yaw_global = np.deg2rad(ego_transform.rotation.yaw)

        if not self.EGO_AUTOPILOT:
            action_point = action.copy()
            if len(self.global_route_ego_frame_no_padding):
                if action[0] in {0, 1, 2} and 1 <= action[1] <= 4:
                    self.index_map = {1: 1, 2: 5, 3: 10, 4: 15}
                    chosen_index = self.index_map[action[1]]

                    # Ensure the chosen index exists in self.global_route_ego_frame
                    if chosen_index >= len(self.global_route_ego_frame_no_padding):
                        chosen_index = len(self.global_route_ego_frame_no_padding) - 1  # Use the last index
                        if chosen_index < 0:
                            chosen_index = 0

                    # If the last index has a negative value, choose index 0
                    if self.global_route_ego_frame_no_padding[chosen_index, 0] < 0:
                        chosen_index = 0

                    action_point = self.global_route_ego_frame_no_padding[chosen_index, :2].copy()

                    if action[0] == 1:
                        action_point[0] = -5.0  # TODO: x position should be perpendicular to the current action
                    elif action[0] == 2:
                        action_point[0] = 5.0   # TODO: x position should be perpendicular to the current action
            else:
                action_point = np.array([0.0, 0.0])

            target_global = self.ego_to_global(np.array(action_point), ego_position_global, ego_yaw_global)
            target_location = carla.Location(x=target_global[0], y=target_global[1], z=current_location.z)

            # Draw the target point in CARLA for debugging
            self.world.debug.draw_point(
                target_location, 
                size=0.1, 
                color=carla.Color(255, 0, 0), 
                life_time=self.FREQUENCY * 2
            )

            # Compute the target speed based on the distance (and REQ_TIME)
            distance = np.linalg.norm(np.array([target_location.x, target_location.y]) - ego_position_global)
            target_speed = distance / self.REQ_TIME * 3.6  # converting to km/h

            # Get the nearest waypoint corresponding to the target location
            target_waypoint = self.world.get_map().get_waypoint(target_location)

            # Calculate control command using the PID controller and apply it
            control = self.pid_controller.run_step(target_speed, target_waypoint)
            self.ego_vehicle.apply_control(control)
        else:
            # Autopilot is enabled, so ignore the provided action.
            pass

        # Advance the simulation by one tick
        self.world.tick()
        self.sim_time += self.FREQUENCY

        # Get new observation from BEV observer
        if self.bev_info.client_init(world=self.world):
            ego_hist, neighbor_hist, map_hist, crosswalk_hist, _ = self.bev_info.update(re_reference=False)
            ego_hist, neighbor_hist, map_hist, ground_truth, crosswalk_hist = \
                self.bev_info.create_buffer_and_transform_frame(np.array(ego_hist),
                                                                np.array(neighbor_hist), 
                                                                np.array(map_hist), 
                                                                np.array(crosswalk_hist))
            ego_obs = self.bev_info.carla_to_MAI_coordinates(data=ego_hist, is_map=False)
            neighbors_obs = self.bev_info.carla_to_MAI_coordinates(data=neighbor_hist, is_map=False)
            map_obs = self.bev_info.carla_to_MAI_coordinates(data=map_hist, is_map=True)
            self.global_route_ego_frame = self.bev_info.carla_to_MAI_coordinates(data=self.global_route_ego_frame,
                                                                                is_map=True)
            if len(ego_obs) > 1:
                ego_obs = ego_obs[-2:-1]
                neighbors_obs = neighbors_obs[-2:-1]
                map_obs = map_obs[-2:-1]

        else:
            ego_obs, neighbors_obs, map_obs = None, None, None

        observation = {"ego": ego_obs, 
                       "neighbors": neighbors_obs, 
                       "map": map_obs,  
                       "global_route": self.global_route_ego_frame}  # New key with the global route.
        self.last_obs = observation  # Save the latest observation for rendering

        # Reward scheme
        reward, done = self._compute_reward(target_location)

        info = {"sim_time": self.sim_time}
        return observation, reward, done, info
    
    def _compute_route_error(self, target_global):
        """
        Given a target point in global coordinates, compute the lateral error and 
        longitudinal progress along the stored global route.
        
        Returns:
            lateral_error (float): Distance from target_global to nearest route point.
            longitudinal_progress (float): Cumulative distance along the route up to that point.
        """
        # Extract the route's (x, y) coordinates.
        route_points = self.global_route[:, :2]  # shape (N, 2)
        
        # Compute the Euclidean distances from the target to each route point.
        target_xy = np.array([target_global.x, target_global.y]).reshape(1, 2)
        distances = np.linalg.norm(route_points - target_xy, axis=1)
        idx = np.argmin(distances)
        lateral_error = distances[idx]
        
        # Compute cumulative distance along the route up to the nearest point.
        if idx == 0:
            longitudinal_progress = 0.0
        else:
            # Sum the distances between successive points along the route.
            longitudinal_progress = np.sum(np.linalg.norm(np.diff(route_points[:idx+1], axis=0), axis=1))
        return lateral_error, longitudinal_progress

    def _get_desired_lane_info(self):
        """
        Determine the desired lane information from the global route based on the ego's current location.
        
        Returns:
            desired_lane_yaw (float): The desired lane direction (yaw in degrees) from the global route.
            desired_lane_point (np.array): The (x, y) coordinate of the closest point on the global route.
        """
        # Get ego's current (x, y) location.
        ego_location = self.ego_vehicle.get_location()
        ego_xy = np.array([ego_location.x, ego_location.y])
        
        # Extract (x, y) coordinates from the global route.
        route_xy = self.global_route[:, :2]  # shape (N, 2)
        
        # Compute distances from ego to each route point.
        distances = np.linalg.norm(route_xy - ego_xy.reshape(1, 2), axis=1)
        idx = np.argmin(distances)
        
        # Get the desired lane yaw from the global route.
        desired_lane_yaw = self.global_route[idx, 2]
        desired_lane_point = route_xy[idx]
        return desired_lane_yaw, desired_lane_point

    def _compute_reward(self, target_global):
        """
        Compute the reward based on:
        - A penalty for being far from the goal.
        - A positive reward for getting closer to the goal.
        - A step penalty to encourage efficiency.
        - No episode termination when the ego reaches the goal.

        Args:
            target_global: A CARLA Location representing the target point in global coordinates.

        Returns:
            reward (float): The computed reward.
            done (bool): Whether the episode should terminate.
        """

        # Terminal condition: If time runs out
        if self.sim_time >= self.SCENE_DURATION:
            return -50.0, True  # Stronger penalty for running out of time
        elif self.collision_detected:
            return -200.0, True  # Ends episode with high penalty for collisions

        # Compute distance to goal
        target_xy = np.array([target_global.x, target_global.y])
        ego_xy = np.array([self.ego_vehicle.get_location().x, self.ego_vehicle.get_location().y])
        distance_to_goal = np.linalg.norm(ego_xy - target_xy)

        # Step penalty (encourages efficiency)
        step_penalty = -1.0

        # New progress reward: Negative when far, positive near the goal
        progress_reward = 100.0 * (1.0 / (1.0 + distance_to_goal) - 1.0)

        # Reward for staying at the goal
        goal_threshold = 0.5  # meters
        if distance_to_goal < goal_threshold:
            progress_reward += 5.0  # Small bonus for staying at the goal

            # Penalize movement after reaching the goal
            # ego_velocity = np.linalg.norm(self.ego_vehicle.get_velocity())  # Compute speed
            # if ego_velocity > 0.1:  # If moving when it should be still
            #     progress_reward -= 2.0  # Small penalty for unnecessary movement

        # Ensure episode doesn't terminate when the goal is reached
        done = False  

        # Total reward
        reward = step_penalty + progress_reward
        return reward, done

    def render(self, mode="human"):
        # Create the renderer if it doesn't already exist.
        if not hasattr(self, 'matplotlib_renderer'):
            self.matplotlib_renderer = MatplotlibAnimationRenderer()

        # Use the stored observation data (if available) to update the renderer.
        if hasattr(self, 'last_obs') and self.last_obs is not None:
            ego_obs = self.last_obs["ego"]
            neighbors_obs = self.last_obs["neighbors"]
            map_obs = self.last_obs["map"]
            self.matplotlib_renderer.update_data(ego_obs, neighbors_obs, map_obs)
    
    def custom_sample_action(self):
        """
        Returns an action that corresponds to FOLLOW_POINT_DIST meters ahead
        of the ego's current position. Since the action is in the ego coordinate system,
        [FOLLOW_POINT_DIST, 0] means x meters forward and 0 lateral offset.
        """
        return np.array([self.FOLLOW_POINT_DIST, 0])

    def close(self):
        """
        Clean up all actors and close the environment.
        """
        self._cleanup()
        if self.render_enabled:
            pygame.quit()


class SaveBestAndManageCallback(BaseCallback):
    """
    Custom callback that:
      - Saves a checkpoint every `save_freq` timesteps.
      - Keeps only the most recent 5 checkpoint files.
      - Uses an EvalCallback to save the best model based on evaluation performance.
    """
    def __init__(self, eval_env, save_freq, save_path, n_eval_episodes=5, verbose=1):
        super(SaveBestAndManageCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.n_eval_episodes = n_eval_episodes
        self.saved_checkpoints = []  # list to store checkpoint filenames
        
        # Create a separate EvalCallback for saving the best model.
        self.eval_callback = EvalCallback(eval_env,
                                          best_model_save_path=save_path,
                                          n_eval_episodes=n_eval_episodes,
                                          verbose=verbose)

    def _on_step(self) -> bool:
        # Periodically save a checkpoint.
        if self.num_timesteps % self.save_freq == 0:
            ckpt_path = os.path.join(self.save_path, f"model_{self.num_timesteps}.zip")
            self.model.save(ckpt_path)
            self.saved_checkpoints.append(ckpt_path)
            if self.verbose > 0:
                print(f"Saved checkpoint: {ckpt_path}")
            # Delete oldest checkpoints if more than 5 are saved.
            if len(self.saved_checkpoints) > 5:
                old_ckpt = self.saved_checkpoints.pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                    if self.verbose > 0:
                        print(f"Deleted old checkpoint: {old_ckpt}")
        return True

    def _on_rollout_end(self):
        # At the end of each rollout, run the evaluation callback.
        self.eval_callback.on_rollout_end()

    def _on_training_end(self):
        # Make sure to call the evaluation callback's training end routine.
        self.eval_callback.on_training_end()

# Example of testing the environment:
if __name__ == '__main__':
    try:
        env = CarlaGymEnv(render_enabled=False)
        eval_env = CarlaGymEnv(render_enabled=RENDER_CAMERA) 
        eval_env.seed(3)

        if TRAIN:
            # Create a directory for saving models/checkpoints.
            save_dir = "./saved_rl_models/1.1"
            os.makedirs(save_dir, exist_ok=True)

            # Create the custom callback: save a checkpoint every 10,000 timesteps.
            callback = SaveBestAndManageCallback(eval_env=eval_env, save_freq=1000, save_path=save_dir, n_eval_episodes=5, verbose=1)

            # Train a small A2C model to confirm everything runs.
            model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
            model.learn(total_timesteps=100_000, callback=callback)

        if TEST:
            # Load the saved model.
            if LOAD_MODEL:
                model = A2C.load("saved_rl_models/model_63000.zip", env=eval_env)
            else:
                model = A2C("MultiInputPolicy", env)
            # Now test with a custom action:
            obs = eval_env.reset()
            done = False
            step_count = 0

            renderer = MatplotlibAnimationRenderer()
            step = 0
            while not done:
                # Selelct random action from env
                # action = eval_env.action_space.sample()   # always drive x meters forward
                # Get action from the trained model
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                step_count += 1
                print(f"Step: {step_count}, Reward: {reward}, Sim Time: {info['sim_time']}")

                # ego_data = obs.get('ego')
                # neighbors_data = obs.get('neighbors')
                # map_data = obs.get('map')
                # route_ef = obs.get('global_route')
                # route_ef = route_ef[route_ef[:, 0] > 0]
                
                # Update the renderer with the latest simulation data.
                # renderer.update_data(ego_data, neighbors_data, map_data, None, route_ef)
                # renderer.update_plot(step)
                step += 1

    except KeyboardInterrupt:
        print("Simulation interrupted.")
    finally:
        env.close()
        print("Environment closed.")

import subprocess
import time

import carla
import gym
import math
from math import sqrt
import pygame
import argparse
from pygame.locals import *
import numpy as np
from gym.utils import seeding

from .hud import HUD
from .world import World
from .vehicle import Vehicle
from .camera import Camera


camera_transforms = {
    "spectator": carla.Transform(carla.Location(x=-10.5, y=2, z=5), carla.Rotation(yaw=-10, pitch=-22)),
    "dashboard": carla.Transform(carla.Location(x=2.9, z=2.5), carla.Rotation(pitch=-45))
}

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

class CarlaEnv(gym.Env):
    """
        heavily adapted from https://github.com/bitsauce/Carla-ppo/tree/master/CarlaEnv
    """

    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, host="127.0.0.1", port=2000,
                 viewer_res=(1280, 720), obs_res=(1280, 720), render=True,
                 synchronous=False, fps=30, action_smoothing=0.9,
                 carla_path=None,):
        """
            Initializes a gym-like environment that can be used to interact with CARLA.
            Connects to a running CARLA enviromment (tested on version 0.9.5) and
            spwans a lincoln mkz2017 passenger car with automatic transmission.
            
            This vehicle can be controlled using the step() function,
            taking an action that consists of [steering_angle, throttle].
            host (string):
                IP address of the CARLA host
            port (short):
                Port used to connect to CARLA
            viewer_res (int, int):
                Resolution of the spectator camera (placed behind the vehicle by default)
                as a (width, height) tuple
            obs_res (int, int):
                Resolution of the observation camera (placed on the dashboard by default)
                as a (width, height) tuple
            action_smoothing:
                Scalar used to smooth the incomming action signal.
                1.0 = max smoothing, 0.0 = no smoothing
            fps (int):
                FPS of the client. If fps <= 0 then use unbounded FPS.
                Note: Sensors will have a tick rate of fps when fps > 0, 
                otherwise they will tick as fast as possible.
            synchronous (bool):
                If True, run in synchronous mode (read the comment above for more info)
            carla_path (str):
                Automatically start CALRA if path to exe is provided.
        """

        # Start CARLA from CARLA_ROOT
        self.carla_process = None
        if carla_path is not None:
            launch_command = [carla_path]
            launch_command += ["Town07"]
            if synchronous: launch_command += ["-benchmark"]
            launch_command += ["-fps=%i" % fps]
            print("Running command:")
            print(" ".join(launch_command))
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.PIPE, universal_newlines=True)
            print("Waiting for CARLA to initialize")
            for line in self.carla_process.stdout:
                if "LogCarla: Number Of Vehicles" in line:
                    break
            time.sleep(2)

        # Initialize pygame for visualization
        width, height = viewer_res
        out_width, out_height = obs_res
        if render:
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()
        else:
            self.clock = None
        self.synchronous = synchronous

        # Setup gym environment
        self.auto_throttle = True
        self.seed()

        if(self.auto_throttle):
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) #just steer
            print("Auto throttle enabled.")
        else:
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) # steer, throttle

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(out_height, out_width, 3), dtype=np.uint8)
        self.metadata["video.frames_per_second"] = self.fps = self.average_fps = fps
        self.spawn_point = 1
        self.action_smoothing = action_smoothing

        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(60.0)

            # Create world wrapper
            self.world = World(self.client)
            for actor in self.world.get_actors():
                if actor.type_id == 'vehicle.tesla.model3':
                    actor.destroy()

            if self.synchronous:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)

            # Get spawn location
            #lap_start_wp = self.world.map.get_waypoint(carla.Location(x=-180.0, y=110))
            lap_start_wp = self.world.map.get_waypoint(self.world.map.get_spawn_points()[1].location)
            self.spawn_transform = lap_start_wp.transform
            self.spawn_transform.location += carla.Location(z=1.0)

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, self.spawn_transform,
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            # Create hud
            if render:
                self.hud = HUD(width, height)
                self.hud.set_vehicle(self.vehicle)
                self.world.on_tick(self.hud.on_world_tick)
            else:
                self.hud = None

            # Create cameras
            self.dashcam = Camera(self.world, out_width, out_height,
                                  transform=camera_transforms["dashboard"],
                                  camera_type="sensor.camera.semantic_segmentation",
                                  color_converter=carla.ColorConverter.CityScapesPalette,
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image(e),
                                  sensor_tick=0.0 if self.synchronous else 1.0/self.fps)
            self.camera  = Camera(self.world, width, height,
                                  transform=camera_transforms["spectator"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e),
                                  sensor_tick=0.0 if self.synchronous else 1.0/self.fps)
        except Exception as e:
            self.close()
            raise e

        # Reset env to set initial state
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, is_training=True):
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        #self.vehicle.control.brake = float(0.0)
        self.vehicle.tick()
        self.vehicle.set_transform(self.spawn_transform)
        self.vehicle.set_simulate_physics(False) # Reset the car's physics

        # Give 2 seconds to reset
        if self.synchronous:
            ticks = 0
            while ticks < self.fps * 2:
                self.world.tick()
                try:
                    self.world.wait_for_tick(seconds=1.0/self.fps + 0.1)
                    ticks += 1
                except:
                    pass
        else:
            time.sleep(2.0)

        self.vehicle.set_simulate_physics(True) # Reset the car's physics

        self.terminal_state = False # Set to True when we want to end episode
        self.closed = False         # Set to True when ESC is pressed
        self.extra_info = []        # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None   # Last received observation
        self.viewer_image = self.viewer_image_buffer = None # Last received image to show in the viewer
        self.start_t = time.time()
        self.step_count = 0
        self.is_training = is_training
        # self.start_waypoint_index = self.current_waypoint_index

        # Metrics
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.laps_completed = 0.0

        # DEBUG: Draw path
        #self._draw_path(life_time=1000.0, skip=10)

        # Return initial observation
        return self.step(None)[0]

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):
        if self.hud is None or self.clock is None:
            return
        # Add metrics to HUD
        self.extra_info.extend([
            f"Reward: {self.last_reward:.2f}",
            "",
            f"Distance traveled: {int(self.distance_traveled)}m",
            f"Avg speed:      {(3.6 * self.speed_accum / self.step_count):7.2f} km/h"
        ])

        # Blit image from spectator camera
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation.shape[:2]
        view_h, view_w = self.viewer_image.shape[:2]
        pos = (view_w - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = [] # Reset extra info list

        # Render to screen
        pygame.display.flip()

        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation

    def step(self, action):
        self.last_reward = 0.0

        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")

        # Asynchronous update logic
        if not self.synchronous and self.clock is not None:
            if self.fps <= 0:
                # Go as fast as possible
                self.clock.tick()
            else:
                # Sleep to keep a steady fps
                self.clock.tick_busy_loop(self.fps)
            
            # Update average fps (for saving recordings)
            if action is not None:
                self.average_fps = self.average_fps * 0.5 + self.clock.get_fps() * 0.5

        # Take action
        if action is not None:
            if(self.auto_throttle):
                steer = action[0]

                self.vehicle.control.throttle = 0.5
                self.vehicle.control.brake = 0
                self.vehicle.control.steer = self.vehicle.control.steer * self.action_smoothing + steer * (1.0-self.action_smoothing)

            else:
                steer, throttle = action # [float(a) for a in action]

                self.vehicle.control.steer = self.vehicle.control.steer * self.action_smoothing + steer * (1.0-self.action_smoothing)
                self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)

                if throttle >= 0:
                    self.vehicle.control.brake = 0
                    self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)
                else:
                    brake = -throttle # because throttle is negative, have to make it positive before adding it to brake
                    self.vehicle.control.throttle = 0
                    self.vehicle.control.brake = self.vehicle.control.brake * self.action_smoothing + brake * (1.0-self.action_smoothing)


        # Tick game
        if self.hud is not None:
            self.hud.tick(self.world, self.clock)
        self.world.tick()

        # Synchronous update logic
        if self.synchronous:
            if self.clock is not None:
                self.clock.tick()
            while True:
                try:
                    self.world.wait_for_tick(seconds=1.0/self.fps + 0.1)
                    break
                except:
                    # Timeouts happen occasionally for some reason, however, they seem to be fine to ignore
                    self.world.tick()

        # Get most recent observation and viewer image
        self.observation = self._get_observation()
        self.viewer_image = self._get_viewer_image()
        # encoded_state = self.encode_state_fn(self)

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # DEBUG: Draw current waypoint
        #self.world.debug.draw_point(self.current_waypoint.transform.location, color=carla.Color(0, 255, 0), life_time=1.0)

        # Calculate distance traveled
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()
        
        # Call reward fn
        self.last_reward += self.reward()

        self.total_reward += self.last_reward
        self.step_count += 1

        # Check for ESC press
        if self.hud is not None:
            pygame.event.pump()
            if pygame.key.get_pressed()[K_ESCAPE]:
                self.close()
                self.terminal_state = True

        self.render() # Render

        return self.observation, self.last_reward, self.terminal_state, { "closed": self.closed }

    def reward(self):
        r = .05
        #r = 10 if self.vehicle.get_velocity() != 0 else 0
        #r = min(abs(self.vehicle.get_velocity().x + self.vehicle.get_velocity().y)*10, 10)

        #note reward has -100 applied for a lane invasion (see self._on_invasion())
        w = self.world.get_map().get_waypoint(self.vehicle.get_location())
        right_lane_waypoint = w.get_right_lane()
        #print("Right lane transform - {}".format(right_lane_waypoint.transform))
        #print("Current transform - {}".format(self.vehicle.get_transform()))
        #if (right_lane_waypoint == carla.LaneType.Driving):
            #print("In the right lane")
        if(right_lane_waypoint is not None):
            r += 1/(1+(self.vehicle.get_location().distance(right_lane_waypoint.transform.location)))
            #print("Distance from right_lane_waypoint == {}".format(dis))
   
        #if car falls off map
        transform = self.vehicle.get_transform()
        if transform.location.z < 0:
            r += -100
            self.terminal_state = True

        return r

    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _on_collision(self, event):
        if self.hud is not None:
            self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        if self.hud is not None:
            self.hud.notification("Crossed line %s" % " and ".join(text))
        
        for lane in lane_types:
            if(str(lane) == 'Broken'): #center lane 
                self.last_reward -= 10
                #print("Center lane crossed")
            else:                       #outer lane -- though a NONE lane does appear sometimes in the center, unsure why or how to avoid
                #print("something else crossed")
                self.last_reward -= 100
                #print("Last Reward = {}".format(self.total_reward))
                self.terminal_state = True

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

if __name__ == "__main__":
    # Example of using CarlaEnv with keyboard controls
    env = CarlaEnv(obs_res=(160, 80))
    action = np.zeros(env.action_space.shape[0])
    # for _ in range(5):
    env.reset(is_training=True)
    while True:
        # Process key inputs
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[K_LEFT] or keys[K_a]:
            action[0] = -0.5
        elif keys[K_RIGHT] or keys[K_d]:
            action[0] = 0.5
        else:
            action[0] = 0.0
        action[0] = np.clip(action[0], -1, 1)
        action[1] = 1.0 if keys[K_UP] or keys[K_w] else 0.0

        # Take action
        obs, _, done, info = env.step(action)
        if info["closed"]: # Check if closed
            exit(0)
        env.render() # Render
        if done: break
    # env.close()
import numpy as np
import pygame
from pygame.locals import *
from carla_env.env import CarlaEnv

if __name__ == "__main__":
	# use the env with keyboard controls for now
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
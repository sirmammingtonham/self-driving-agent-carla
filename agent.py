import numpy as np
import pygame
from pygame.locals import *
from carla_env.env import CarlaEnv
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env

def test():
    # use the env with keyboard controls for now
    env = CarlaEnv(obs_res=(160, 160))
    action = np.zeros(env.action_space.shape[0])
    obs = env.reset(is_training=True)

    check_env(env)

    breuh = obs.shape
    while True:
        # print(obs)
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
        assert(breuh == obs.shape)
        if info["closed"]: # Check if closed
            exit(0)
        env.render() # Render
        if done: break
    env.close()

def train_ppo():
    env = CarlaEnv(obs_res=(160, 160), render=False)
    check_env(env)
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=int(2e5))
    # Save the agent
    model.save("carla_test")

if __name__ == "__main__":
	train_ppo()
    # test()
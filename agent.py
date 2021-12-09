import numpy as np
import pygame
import sys
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

def train_ppo(renderMode=False):
    env = CarlaEnv(obs_res=(160, 160), render=renderMode)
    check_env(env)
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
    model.learn(total_timesteps=int(2e5), tb_log_name="first_run")
    # Save the agent
    model.save("carla_test")

def load_ppo(filepath, renderMode=False):
    env = CarlaEnv(obs_res=(160,160), render=renderMode)
    check_env(env)

    model = PPO.load(filepath, env=env)

    for i in range(1000):
        action = env.action_space.sample()
        observation, last_reward, terminal_state, closed = env.step(action)
        if(terminal_state):
            env.reset()

#this code reads in commandline arguments, then calls the appropriate environment setup.
# -train sets to training mode
# -load [filepath] loads a saved model at filepath. including '.zip' is optional
# -test sets to testing mode
# -render enables rendering the agent's perspective an a HUD with more information
# default is to train without separate rendering
if __name__ == "__main__":
    args = sys.argv[1:]

    if(len(args) == 0):
        train_ppo(renderMode=False)
    else:
        mode = None
        render = False
        for (index, arg) in enumerate(args):
            if(mode == None and arg == '-train'):
                mode = 'train'
            elif(mode == None and arg == '-test'):
                mode = 'test'
            elif(mode == None and arg == '-load'):
                if(index == len(args)-1):
                    print('ERROR: no filepath provided for loading model')
                    mode = 'error'
                else:
                    mode = args[index+1]
                    del args[index+1]
            elif(arg == '-render'):
                print('Agent will render.')
                render = True
            else:
                print('WARNING: argument %s invalid.' % (arg))
                

        if(mode == 'error'):
            pass
        elif(mode == 'train' or mode == None):
            print('Training model.')
            train_ppo(render)
        elif(mode == 'test'):
            print('Entering test mode')
            test()
        else:
            print('Loading model at filepath %s.' % (mode))
            load_ppo(mode, render)
	
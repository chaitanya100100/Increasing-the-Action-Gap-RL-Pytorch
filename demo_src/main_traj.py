import gym
import torch.optim as optim

from dqn_model import DQN
from dqn_learn_traj import run_episode
from utils.gym_file import get_env, get_wrapper_by_name
import numpy as np
import pickle

REPLAY_BUFFER_SIZE = 30000
FRAME_HISTORY_LEN = 4

game = "SpaceInvaders"

def main(env):


    all_obs = run_episode(
        env=env,
        q_func=DQN,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        frame_history_len=FRAME_HISTORY_LEN,
        game=game,
    )

    with open("Traj_{}.npy".format(game), "wb") as f:
        pickle.dump(all_obs, f)
    

if __name__ == '__main__':


    benchmark = gym.benchmark_spec('Atari40M')


    #task = 'AsterixNoFrameskip-v4'
    task = "{}NoFrameskip-v4".format(game)

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)



    #print(task.max_timesteps)
    main(env)

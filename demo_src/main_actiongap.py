import gym
import torch.optim as optim

from dqn_model import DQN
from dqn_learn_actiongap import run_episode
from utils.gym_file import get_env, get_wrapper_by_name
import numpy as np
import pickle


FRAME_HISTORY_LEN = 4

game = "Breakout"

def main(env):


    obss = np.load("Traj_{}.npy".format(game))

    DQN_gaps = run_episode(
        env=env,
        q_func=DQN,
        frame_history_len=FRAME_HISTORY_LEN,
        oper = "DQN",
        game=game,
        obss=obss,
    )

    AL_gaps = run_episode(
        env=env,
        q_func=DQN,
        frame_history_len=FRAME_HISTORY_LEN,
        oper = "AL",
        game=game,
        obss=obss,
    )

    PAL_gaps = run_episode(
        env=env,
        q_func=DQN,
        frame_history_len=FRAME_HISTORY_LEN,
        oper = "PAL",
        game=game,
        obss=obss,
    )

    with open("AG_{}.pkl".format(game), "wb") as f:
        pickle.dump({'DQN_gaps':DQN_gaps, 'AL_gaps':AL_gaps, 'PAL_gaps':PAL_gaps}, f)
    

if __name__ == '__main__':


    benchmark = gym.benchmark_spec('Atari40M')


    #task = benchmark.tasks[3]
    #task = 'AsterixNoFrameskip-v4'
    task = "{}NoFrameskip-v4".format(game)

    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)



    #print(task.max_timesteps)
    main(env)

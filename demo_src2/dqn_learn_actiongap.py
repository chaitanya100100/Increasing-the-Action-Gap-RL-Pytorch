import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import os

from utils.replay_buffer import ReplayBuffer
from utils.gym_file import get_wrapper_by_name


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)



def run_episode(
    env,
    q_func,
    frame_history_len,
    oper,
    game,
    obss,
    ):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete


    if len(env.observation_space.shape) == 1:
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
    num_actions = env.action_space.n


    Q = q_func(input_arg, num_actions).type(dtype)
    Q.load_state_dict(torch.load("./models/{}_{}.pth".format(oper, game)))


    gaps = []


    for obs in obss:

        torch_obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
        with torch.no_grad():
            Qvals = Q(torch_obs).data[0]
        max2val, max2idx = Qvals.topk(2)
        gaps.append((max2val[0]-max2val[1]).item())

    
    return gaps

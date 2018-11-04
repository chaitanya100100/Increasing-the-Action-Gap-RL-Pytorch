from model import DQN
import torch
import torch.optim as optim

import numpy as np
import random

from pong import Pong
from memory import MemoryReplay

import time
from utils import (sample_action, save_statistic)
from collections import deque


VALID_ACTION = [0, 3, 4]
GAMMA = 0.99
epsilon = 0.5
update_step = 1000
memory_size = 2000
max_epoch = 100000
batch_size = 64
K = 4 # Number of frame skips
save_path = './tmp'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Variables
var_phi = torch.zeros((1, 4, 84, 84)).to(device)

# For training
var_batch_phi = torch.zeros((batch_size, 4, 84, 84), requires_grad=True).to(device)
var_batch_a = torch.zeros((batch_size, 1), dtype=torch.long, requires_grad=False).to(device)
var_batch_r = torch.zeros((batch_size, 1), requires_grad=True).to(device)
var_batch_phi_next = torch.zeros((batch_size, 4, 84, 84), requires_grad=True).to(device)
var_batch_r_mask = torch.zeros((batch_size, 1), requires_grad=False).to(device)


MP = MemoryReplay(memory_size, batch_size)
dqn = DQN().to(device)
target_dqn = DQN().to(device)
target_dqn.load_state_dict(dqn.state_dict())


optimz = optim.RMSprop(dqn.parameters(),  lr=0.0025, alpha=0.9, eps=1e-02, momentum=0.0)

pong = Pong()

for i in range(memory_size):
	phi = pong.current_phi
	act_index = random.randrange(3)
	phi_next, r, done = pong.step(VALID_ACTION[act_index])
	pong.display()
	MP.put((phi_next, act_index, r, done))

	if done:
		pong.reset()

print("================\n"
	  "Start training!!\n"
	  "================")
pong.reset()

epoch = 0
update_count = 0
score = 0.
avg_score = -21.0
best_score = -21.0

t = time.time()

SCORE = []
QVALUE = []
QVALUE_MEAN = []
QVALUE_STD = []

while(epoch < max_epoch):

	cnt = 0
	while(not done):

		optimz.zero_grad()

		act_index = sample_action(pong, dqn, var_phi, epsilon)

		epsilon = (epsilon - 1e-6) if epsilon > 0.1 else  0.1

		for _ in range(K):
			phi_next, r, done = pong.step(VALID_ACTION[act_index])
			if _ == 0:
				MP.put((phi_next, act_index, r, done))
			r = np.clip(r, -1, 1)
			score += r

		pong.display()

		# batch sample from memory to train
		batch_phi, batch_a, batch_r, batch_phi_next, batch_done = MP.batch()
		var_batch_phi_next.data.copy_(torch.from_numpy(batch_phi_next))
		batch_target_q, _ = target_dqn(var_batch_phi_next).max(dim=1)

		mask_index = np.ones((batch_size, 1))
		mask_index[batch_done] = 0.0
		var_batch_r_mask.data.copy_(torch.from_numpy(mask_index))

		var_batch_r.data.copy_(torch.from_numpy(batch_r).unsqueeze(1))

		y = var_batch_r + batch_target_q.mul(GAMMA).mul(var_batch_r_mask)
		y = y.detach()

		var_batch_phi.data.copy_(torch.from_numpy(batch_phi))
		batch_q = dqn(var_batch_phi)

		var_batch_a.data.copy_(torch.from_numpy(batch_a).long().view(-1, 1))
		batch_q = batch_q.gather(1, var_batch_a)

		loss = y.sub(batch_q).pow(2).mean()
		loss.backward()
		optimz.step()

		update_count += 1
		cnt += 1

		if update_count == update_step:
			target_dqn.load_state_dict(dqn.state_dict())
			update_count = 0

		QVALUE.append(batch_q.data.cpu().numpy().mean())

	SCORE.append(score)
	QVALUE_MEAN.append(np.mean(QVALUE))
	QVALUE_STD.append(np.std(QVALUE))
	QVALUE = []

	save_statistic('Score', SCORE, save_path=save_path)
	save_statistic('Average Action Value', QVALUE_MEAN, QVALUE_STD, save_path)

	pong.reset()
	done = False
	epoch += 1
	avg_score = 0.9*avg_score + 0.1*score
	score = 0.0
	print('Epoch: {0}. Steps: {1}. Avg.Score:{2:6f}'.format(epoch, cnt, avg_score))

	time_elapse = time.time() - t

	if avg_score >= best_score and time_elapse > 300:
		torch.save(dqn.state_dict(), save_path+'/model.pth')
		print('Model has been saved.')
		best_score = avg_score
		t = time.time()

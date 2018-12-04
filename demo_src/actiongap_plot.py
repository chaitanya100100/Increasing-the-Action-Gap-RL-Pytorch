import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

game = "Pong"


x = pkl.load(open("./AG_{}.pkl".format(game), "rb"))

x['DQN_gaps'] = np.array(x['DQN_gaps']).clip(0.0, 0.1)[::2]#**0.5
x['AL_gaps'] = np.array(x['AL_gaps']).clip(0.0, 0.1)[::2]#**0.5
x['PAL_gaps'] = np.array(x['PAL_gaps']).clip(0.0, 0.1)[::2]#**0.5


plt.plot(x['DQN_gaps'], label="DQN")
plt.plot(x['AL_gaps'], label="AL")
plt.plot(x['PAL_gaps'], label="PAL")

plt.title("{}".format(game))
plt.xlabel("episode step")
plt.ylabel("action gap")
plt.legend()
plt.show()

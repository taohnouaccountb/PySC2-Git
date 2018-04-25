import json
import numpy as np
import sys
import os

for filepath in sys.argv[1:]:
    filedir, filename = os.path.split(filepath)
    filename = filename[:-5]
    targetpath = filepath[:-5]+'.npz'
    print(targetpath)
    x = json.load(open(filepath))
    np.savez(targetpath, episode_reward=x['episode_reward'],
             nb_episode_steps=x['nb_episode_steps'], nb_steps=x['nb_steps'])

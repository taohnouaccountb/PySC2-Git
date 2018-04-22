from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import time
import os

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

COLLECT_SIZE = 2000

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_UNIT = actions.FUNCTIONS.Move_screen.id
_ATTACK_UNIT = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
ACTION_OPTIONS = [_NO_OP, _MOVE_UNIT, _ATTACK_UNIT, _SELECT_ARMY]

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_UNIT_ENERGY = features.SCREEN_FEATURES.unit_energy.index
_UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index
SCREEN_FEATURES = [i.index for i in features.SCREEN_FEATURES]


class RandomAgent(base_agent.BaseAgent):
    """A random agent of DefeatZerglingsAndBanelings"""

    def __init__(self):
        super(RandomAgent, self).__init__()
        self.state_pool = []
        self.action_pool = []
        self.action_axis_pool = []
        self.reward_pool = []
        self.nstate_pool = []
        self.r_state = None
        self.r_action = None
        self.r_reward = None
        self.first = True
        self.count = 0
        self.PATH = 'E:\\cse496_hw2\\PySC2-Git\\data\\OUTPUT\\'
        self.reset_flag = 0
        self.

    def setup(self, obs_spec, action_spec):
        super(RandomAgent, self).setup(obs_spec, action_spec)
        ltime = time.localtime()
        dir = str(ltime.tm_mon) + 'm' + str(ltime.tm_mday) + 'd'
        dir += str(ltime.tm_hour) + 'h' + str(ltime.tm_min) + 'm' + str(ltime.tm_sec) + 's'
        self.PATH += dir + "\\"
        os.mkdir(self.PATH)


    def smart_store(self, state, action, reward, nstate):
        self.state_pool.append(state)
        self.action_pool.append(action[0])
        if len(action) == 2:
            x = action[1]
            axis = [x, x]
        elif len(action) == 3:
            x = action[1]
            y = action[2]
            axis = [x, y]
        else:
            axis = [-1, -1]
        self.action_axis_pool.append(axis)
        self.reward_pool.append(reward)
        self.nstate_pool.append(nstate)

        if len(self.reward_pool) == COLLECT_SIZE:
            self.save_to_file(self.PATH + 'ep_test_' + str(self.count))
            self.state_pool = []
            self.action_pool = []
            self.action_axis_pool = []
            self.reward_pool = []
            self.nstate_pool = []
            self.count = self.count + 1
            if self.count == 10:
                sys.exit()

    def save_to_file(self, prefix):
        np.savez(prefix, state=self.state_pool,
                 action=self.action_pool,
                 action_axis=self.action_axis_pool,
                 reward=self.reward_pool,
                 nstate=self.nstate_pool)

    def step(self, obs):
        # if self.reset_flag == 0:
        #     pass
        # elif self.reset_flag


        # simulate
        super(RandomAgent, self).step(obs)
        avail_actions = obs.observation["available_actions"]
        avail_action_ids = list(filter(lambda x: x in avail_actions, ACTION_OPTIONS))
        action_id = avail_action_ids[np.random.randint(0, len(avail_action_ids))]
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[action_id].args]
        ret = actions.FunctionCall(action_id, args)
        # record
        screen = [obs.observation["screen"][i] for i in SCREEN_FEATURES]
        action = [action_id, ]
        action.extend(args)

        nstate = screen
        if not self.first:
            self.smart_store(self.r_state, self.r_action, self.r_reward, nstate)
        else:
            self.first = False

        self.r_state = nstate
        self.r_action = action
        self.r_reward = obs.reward
        return ret

    def reset(self):
        super(RandomAgent, self).reset()
        self.first = True

    # def translate_to_pysc2(self, a):
    #     pass
    #
    # def translate_from_pysc2(self, a):
    #     if a[0] == _NO_OP:
    #         return 0
    #     elif a[0] == _SELECT_ARMY

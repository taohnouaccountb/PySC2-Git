from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import time
import os
from collections import deque
from copy import copy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

ADD_SELECT_ARMY = False

COLLECT_SIZE = 3000

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
ACTION_OPTIONS = [_NO_OP, _MOVE_SCREEN, _ATTACK_SCREEN, _SELECT_ARMY]

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_HIT_POINTS = features.SCREEN_FEATURES.unit_hit_points.index
_UNIT_ENERGY = features.SCREEN_FEATURES.unit_energy.index
_UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index
SCREEN_FEATURES = [_PLAYER_RELATIVE, _UNIT_TYPE, _SELECTED, _UNIT_DENSITY, _UNIT_ENERGY]

def number_of_channels():
    return len(SCREEN_FEATURES)

class Action(object):
    """ Action Containiner including translator """

    def __init__(self, code):
        # [_NO_OP, _SELECT_ARMY, _MOVE_SCREEN*_MOVE_SCREEN, _ATTACK_SCREEN*_ATTACK_SCREEN]
        self.code = code

        self._screen_size = 64
        self._move_size = 10
        self._attack_size = 10

        self._no_op = 0
        
        if ADD_SELECT_ARMY:
            self._select_army = 1
            self._move_lower_bound = self._select_army+1
        else:
            self._move_lower_bound = self._no_op+1

        self._move_upper_bound = self._move_lower_bound+self._move_size*self._move_size

        self._attack_lower_bound = self._move_upper_bound
        self._attack_upper_bound = self._attack_lower_bound+self._attack_size*self._attack_size

        self._action_space = self._attack_upper_bound


    def to_pysc2(self):
        if self.code == self._no_op:
            return (0,[])
        elif ADD_SELECT_ARMY and self.code == self._select_army:
            return (_SELECT_ARMY,[[0]])
        elif self.code >= self._move_lower_bound and self.code < self._move_upper_bound:
            delta = self.code - self._move_lower_bound
            x, y = self.axis_mapping(delta, self._move_size)
            return (_MOVE_SCREEN, [[0], [x,y]])
        elif self.code >= self._attack_lower_bound and self.code < self._attack_upper_bound:
            delta = self.code - self._attack_lower_bound
            x, y = self.axis_mapping(delta, self._attack_size)
            return (_ATTACK_SCREEN, [[0], [x,y]])
        else:
            raise Exception('Bad Action Code in random_gen.Action.to_pysc2')

    def axis_mapping(self, delta, scale):
        y = delta % scale
        x = (delta - y) / scale
        size = self._screen_size - 1
        x = int(x*1.0/(scale-1)*size)
        y = int(y*1.0/(scale-1)*size)
        return x, y

    @staticmethod
    def get_size():
        x = Action(0)
        return x._action_space

    @staticmethod
    def random_gen():
        x = Action(0)
        size = x._action_space
        return Action(np.random.randint(0, size))

class TupleMaker(object):
    """ Simulate the phi function in the Atari paper. """

    def __init__(self):
        self._memory_length = 4

        self.count = 0
        self.q = None
        self.s_action = None
        self.s_reward = None

        self.tuple = None

    def update(self, state, action, reward):
        self.count = self.count + 1
        # print('TUPLE COUNT:',self.count)
        if(self.count == 1):
            self.q = deque([])
            while len(self.q)<self._memory_length:
                self.q.append(state)
        else:
            pre_state = copy(self.q)
            self.q.popleft()
            self.q.append(state)
            self.tuple = self.pack(pre_state)

        self.s_action = action
        self.s_reward = reward
        return

    def get(self):
        return self.tuple

    def pack(self, state):
        state = self.vstack(state)
        nstate = self.vstack(self.q)
        # state action reward nstate
        return (state, self.s_action, self.s_reward, nstate)

    def vstack(self, que):
        return np.vstack(que)

class RandomAgent(base_agent.BaseAgent):
    """A random agent of DefeatZerglingsAndBanelings"""

    def __init__(self):
        super(RandomAgent, self).__init__()
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.nstate_pool = []
        self.end_pool = []
        self.r_state = None
        self.r_action = None
        self.r_reward = None
        self.count = 0
        self.PATH = 'E:\\cse496_hw2\\PySC2-Git\\data\\OUTPUT\\'
        self.apm_count = 0
        self.tuple = None

    def setup(self, obs_spec, action_spec):
        super(RandomAgent, self).setup(obs_spec, action_spec)
        ltime = time.localtime()
        dir = str(ltime.tm_mon) + 'm' + str(ltime.tm_mday) + 'd'
        dir += str(ltime.tm_hour) + 'h' + str(ltime.tm_min) + 'm' + str(ltime.tm_sec) + 's'
        self.PATH += dir + "\\"
        os.mkdir(self.PATH)

    def step(self, obs):
        # APM Controle
        self.apm_count = self.apm_count+1
        if not self.apm_count%2==0:
            return actions.FunctionCall(0,[])

        # Make action
        op = Action.random_gen()
        action_id, args = op.to_pysc2()
        action_avil = obs.observation["available_actions"]
        if not action_id in action_avil:
            action_id = _NO_OP
            args = []
            op.code = 0
        op_env = actions.FunctionCall(action_id, args)

        # Save tuple
        state = [obs.observation["screen"][i] for i in SCREEN_FEATURES]
        self.tuple.update(state, op.code, obs.reward)
        # print('AGENT COUNT:',self.apm_count)
        if not self.tuple.get() is None:
            self.smart_store()

        # print((action_id,args))
        return op_env

    def reset(self):
        super(RandomAgent, self).reset()
        if not self.apm_count == 0:
            self.end_pool.pop()
            self.end_pool.append(1)
        self.apm_count = 0
        self.tuple = TupleMaker()

    def smart_store(self):
        state, action, reward, nstate = self.tuple.get()

        self.state_pool.append(state)
        self.action_pool.append(action)
        self.reward_pool.append(reward)
        self.nstate_pool.append(nstate)
        self.end_pool.append(0)

        if len(self.reward_pool) == COLLECT_SIZE:
            self.save_to_file(self.PATH + 'ep_test_' + str(self.count))
            self.state_pool = []
            self.action_pool = []
            self.reward_pool = []
            self.nstate_pool = []
            self.end_pool = []
            self.count = self.count + 1
            if self.count == 5:
                sys.exit()

    def save_to_file(self, prefix):
        pass
        np.savez(prefix, state=self.state_pool,
                 action=self.action_pool,
                 reward=self.reward_pool,
                 nstate=self.nstate_pool,
                 end=self.end_pool)
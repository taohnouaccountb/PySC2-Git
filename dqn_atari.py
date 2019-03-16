from __future__ import division

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from pysc2.agents.random_gen import Action, number_of_channels, SCREEN_FEATURES
from pysc2.env.sc2_env import SC2Env
from pysc2.lib import actions

import tensorflow as tf
from keras import backend as K
INPUT_SHAPE = (64, 64,)
WINDOW_LENGTH = 4
GPU_ID = '1'

config = tf.ConfigProto()
config.gpu_options.visible_device_list = GPU_ID
K.set_session(tf.Session(config=config))


# Revised from Keras-RL Github
class SC2Processor(Processor):

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.
        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.
        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            observation (object): An observation as obtained by the environment
        # Returns
            Observation obtained by the environment processed
        """
        # observation = observation[0].observation
        observation = np.array([observation["screen"][i] for i in SCREEN_FEATURES])

        assert observation.ndim == 3  # (channel, height, width)
        processed_observation = observation
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            reward (float): A reward as obtained by the environment
        # Returns
            Reward obtained by the environment processed
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        # Arguments
            info (dict): An info as obtained by the environment
        # Returns
            Info obtained by the environment processed
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.
        # Arguments
            action (int): Action given to the environment
        # Returns
            Processed action given to the environment
        """
        op = Action(action)
        action_id, action_args = op.to_pysc2()
        op_env = actions.FunctionCall(action_id, action_args)
        return [op_env]

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        # Arguments
            batch (list): List of states
        # Returns
            Processed list of states
        """
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        # processed_batch = batch.astype('float32') / 255.
        processed_batch = batch.astype('float32')
        return processed_batch


class Args(object):
    def __init__(self):
        self.mode = 'train'
        self.env_name = 'DefeatZerglingsAndBanelings'


if __name__ == '__main__':

    args = Args()

    # Get the environment and extract the number of actions.
    np.random.seed(123)
    nb_actions = Action.get_size()
    env = SC2Env(map_name=args.env_name, visualize=False)

    # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
    input_shape = (WINDOW_LENGTH, number_of_channels(),) + INPUT_SHAPE
    model = Sequential()
    model.add(Permute((3, 4, 1, 2), input_shape=input_shape))
    model.add(Reshape(INPUT_SHAPE + (WINDOW_LENGTH * number_of_channels(),)))
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=300000, window_length=WINDOW_LENGTH)
    processor = SC2Processor()

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=2000000)

    # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # is Boltzmann-style exploration:
    # policy = BoltzmannQPolicy(tau=1.)
    # Feel free to give it a try!

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    if args.mode == 'train':
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
        weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
        checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
        log_filename = 'dqn_{}_log.json'.format(args.env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=50000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        dqn.fit(env, callbacks=callbacks, nb_steps=2000000, log_interval=10000)

        # After training is done, we save the final weights one more time.
        dqn.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        dqn.test(env, nb_episodes=10, visualize=False)
    elif args.mode == 'test':
        weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
        if args.weights:
            weights_filename = args.weights
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=10, visualize=True)

import numpy as np
from rl.core import Env, Space
from pysc2.agents.random_gen import Action


class ActionSpace(Space):
    """Abstract model for a space that is used for the state and action spaces. This class has the
    exact same API that OpenAI Gym uses so that integrating with it is trivial.

    Please refer to [Gym Documentation](https://gym.openai.com/docs/#spaces)
    """
    def __init__(self):
        super(ActionSpace, self).__init__()
        x = Action()
        self.size = x._action_space

    def sample(self, seed=None):
        """Uniformly randomly sample a random element of this space.
        """
        return np.random.randint(self.size)

    def contains(self, x):
        """Return boolean specifying if x is a valid member of this space
        """
        return x < self.size


class Sc2Env(Env):
    """The abstract environment class that is used by all agents. This class has the exact
    same API that OpenAI Gym uses so that integrating with it is trivial. In contrast to the
    OpenAI Gym implementation, this class only defines the abstract methods without any actual
    implementation.

    To implement your own environment, you need to define the following methods:

    - `step`
    - `reset`
    - `render`
    - `close`

    Refer to the [Gym documentation](https://gym.openai.com/docs/#environments).
    """
    reward_range = (0, 1)
    action_space = ActionSpace()
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        raise NotImplementedError()

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.

        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        raise NotImplementedError()

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        raise NotImplementedError()

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        raise NotImplementedError()

    def configure(self, *args, **kwargs):
        """Provides runtime configuration to the environment.
        This configuration should consist of data that tells your
        environment how to run (such as an address of a remote server,
        or path to your ImageNet data). It should not affect the
        semantics of the environment.
        """
        raise NotImplementedError()

    def __del__(self):
        self.close()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)




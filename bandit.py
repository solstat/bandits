"""

"""

import numpy as np
from arm import *


class Bandit:
    def __init__(self, **kwargs):
        raise NotImplementedError

    @property
    def num_arms(self):
        raise NotImplementedError

    @property
    def arm_names(self):
        """ The arms in the bandit

                Returns:
                    a list of the arms in the bandit. These
                    should be hashable and have a str representation.
        """
        raise NotImplementedError

    def pull_arm(self, arm_index):
        """ Pulls an arm of the bandit

                Returns:
                    a double corresponding to the reward of pulling the
                    arm of the bandit.
        """
        raise NotImplementedError


class StaticBandit(Bandit):
    def __init__(self, arms):
        self._arms = arms
        self._num_arms = len(arms)
        self._arm_names = [arm.name for arm in arms]

    @property
    def num_arms(self):
        return self._num_arms

    @property
    def arm_names(self):
        return self._arm_names

    def pull_arm(self, arm_index):
        return self._arms[arm_index].pull()


class DynamicBandit(Bandit):
    def __init__(self, arms):
        self.__arms = arms
        self.__num_arms = len(arms)
        self.__arm_names = [arm.name for arm in arms]

    @property
    def num_arms(self):
        return self.__num_arms

    @property
    def arm_names(self):
        return self.__arm_names

    def pull_arm(self, arm_index):
        observed_reward = np.nan
        for i in range(self.num_arms):
            reward = self.__arms[i].pull()
            if i == arm_index:
                observed_reward = reward
        return observed_reward


# TODO: noise_func isnt defined in LinearInterpolationArm constructor. pls fix?
class LinearInterpolationBandit(DynamicBandit):
    """ Linear interpolation bandit
    
    Uses Gaussain with standard deviation 1.0
    
    Args:
        means (num_arms by num_periods array): 
            mean score for each arm for each period
        periods (num_periods array): 
            number of pull between periods
        iteration 
    """    
    def __init__(self, means = np.array([[0,10], [10,0]]),
                 periods = [100, 100],
                 iteration = 0, noise_func=None, **kwargs):
        arms = [LinearInterpolationArm(means[i,:], periods, iteration, noise_func=noise_func) for i in range(len(means))]
        DynamicBandit.__init__(self, arms)
        return

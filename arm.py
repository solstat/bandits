"""

This file contains Arm interface and its implemented classes.

Users (typically Bandits) interact with Arms through the pull() method, which:
    - returns a reward value
    - advances state of the arm's parameters (if valid)

"""

import numpy as np

class Arm:
    """ A bandit arm

        Should keep track of the internal state of the arm
        and return the appropriate reward when pulled.
    """

    def __init__(self, **kwargs):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def pull(self):
        """ Pull the arm

            Pulls the bandit arm, returns a double representing the
            award, and advances internal state (if any).
        """
        raise NotImplementedError


class WhiteNoiseArm(Arm):
    def __init__(self, name, rng):
        self.__name = name
        self._rng = rng

    @property
    def name(self):
        return self.__name

    def pull(self):
        return self._rng()


class BernoulliArm(WhiteNoiseArm):
    """Generates iid observations from a Bernoulli white noise"""
    def __init__(self, prob):
        WhiteNoiseArm.__init__(self,
                               name='bernoulli_arm',
                               rng=lambda: np.random.binomial(n=1, p=prob))
        self.prob = prob


class GaussianArm(WhiteNoiseArm):
    """Generates iid observations from a Gaussian white noise"""
    def __init__(self, mu, sigma):
        WhiteNoiseArm.__init__(self,
                               name='gaussian_arm',
                               rng=lambda: np.random.normal(loc=mu, scale=sigma))
        self.mu = mu
        self.sigma = sigma


class LinearInterpolationArm(Arm):
    """ Linear interpolation arm
    """

    def __init__(self, means, periods, iteration):
        self.__name = "lin_interp_arm"

        self.num_periods = len(means)
        self.means = means
        self.iteration = iteration
        self.periods = periods

        if np.size(periods) != self.num_periods:
            raise ValueError("periods not correct size")

        return

    @property
    def name(self):
        return self.__name

    def pull(self):
        iter_to_end_period = self.iteration % np.sum(self.periods)
        end_period = 0
        while (iter_to_end_period >= 0):
            iter_to_end_period -= self.periods[end_period]
            end_period += 1
        start_period = end_period - 1
        end_period = end_period % self.num_periods
        start_frac = np.abs(iter_to_end_period) / self.periods[start_period]

        arm_mean = (
            start_frac * self.means[start_period] +
            (1.0 - start_frac) * self.means[end_period]
        )

        reward = np.random.normal(loc=arm_mean)
        self.iteration += 1

        return reward

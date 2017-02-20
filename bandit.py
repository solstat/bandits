import numpy as np

class Bandit:
    def __init__(self):
        raise NotImplementedError

    def get_arms_list(self):
        """ The arms in the bandit

                Returns:
                    a list of the arms in the bandit. These
                    should be hashable and have a str representation.
        """
        raise NotImplementedError

    def pull_arm(self, arm):
        """ Pulls an arm of the ba

                Returns:
                    a double corresponding to the reward of pulling the
                    arm of the bandit.
        """
        raise NotImplementedError


class StaticBernoulli(Bandit):
    def __init__(self, probs):
        self.probs = probs

    def get_arms_list(self):
        return range(len(self.probs))

    def pull_arm(self, arm):
        return np.random.binomial(n=1, p=self.probs[arm])


class StaticGaussian(Bandit):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def get_arms_list(self):
        return range(len(self.mu))

    def pull_arm(self, arm):
        return np.random.normal(loc=self.mu[arm], scale=self.sigma[arm])



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
        return self.__arms[arm_index].pull()


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


# class StaticBernoulli(Bandit):
#     def __init__(self, probs):
#         self.probs = probs
#
#     def arm_names(self):
#         return range(len(self.probs))
#
#     def pull_arm(self, arm):
#         return np.random.binomial(n=1, p=self.probs[arm])
#
#
# class StaticGaussian(Bandit):
#     def __init__(self, mu, sigma):
#         self.mu = mu
#         self.sigma = sigma
#
#     def arm_names(self):
#         return range(len(self.mu))
#
#     def pull_arm(self, arm):
#         return np.random.normal(loc=self.mu[arm], scale=self.sigma[arm])

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
                 iteration = 0):
        arms = [LinearInterpolationArm(means[i,:], periods, iteration) for i in range(len(means))]
        super(LinearInterpolationBandit, self).__init__(arms)
        return

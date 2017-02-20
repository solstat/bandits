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



class DynamicBandit(Bandit):
    """ Dynamic Bandit 
    
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

        self._num_arms = np.shape(means)[0]
        self._num_periods = np.shape(means)[1]
        self.means = means
        self.iteration = iteration
        
        if np.size(periods) != self._num_periods:
            raise ValueError("periods not correct size")
        self.periods = periods
        return

    def get_arms_list(self):
        """ The arms in the bandit

                Returns:
                    a list of the arms in the bandit. These
                    should be hashable and have a str representation.
        """
        arms_list = np.arange(self._num_arms, dtype = int)
        return arms_list.tolist()

    def pull_arm(self, arm):
        """ Pulls an arm of the bandit

                Returns:
                    a double corresponding to the reward of pulling the
                    arm of the bandit.
        """
        # Find Current place between two periods
        iter_to_end_period = self.iteration % np.sum(self.periods)
        end_period = 0
        while(iter_to_end_period >= 0):
            iter_to_end_period -= self.periods[end_period]
            end_period += 1
        start_period = end_period - 1
        end_period = end_period % self._num_periods
        start_frac = np.abs(iter_to_end_period) / self.periods[start_period]
        
        arm_mean = ( 
                start_frac * self.means[arm, start_period] + 
                (1.0 - start_frac) * self.means[arm, end_period]
                )
        
        reward = np.random.normal(loc=arm_mean)
        self.iteration += 1
        
        return reward

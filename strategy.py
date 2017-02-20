import numpy as np
from scipy.stats import beta

class Strategy:
    """ Base Strategy Class

    Args:

    Attributes:


    """

    def __init__(self, bandit, **kwargs):
        raise NotImplementedError

    def fit(self, iterations, **kwargs):
        """ Fit

        Args:
            iterations (int): number of lsjdflajdf
            **kwargs: other

        Returns:
            A dictionary with arguments:
                rewards (list): the values returned by the bandit
                                at every iteration.
                arms_pulled (list): the arm pulled at every iteration.
                ...
        """
        raise NotImplementedError


class ThompsonBernoulli(Strategy):
    def __init__(self, bandit, alpha_prior, beta_prior):
        self.bandit = bandit
        self.num_arms = len(bandit.get_arms_list())
        self.alpha_posterior = [alpha_prior] * self.num_arms
        self.beta_posterior = [beta_prior] * self.num_arms

    def choose_arm(self):
        mean_samples = [beta.rvs(a, b, size=1) for a, b in
                        zip(self.alpha_posterior, self.beta_posterior)]
        return np.argmax(mean_samples)

    def pull_arm(self, arm_index):
        return self.bandit.pull_arm(arm_index)

    def update_mean_posterior(self, arm_index, observed_reward):
        if observed_reward == 1:
            self.alpha_posterior[arm_index] += 1
        else:
            self.beta_posterior[arm_index] += 1

    def fit(self, iterations):
        for iter in range(iterations):
            arm_index = self.choose_arm()
            observed_reward = self.pull_arm(arm_index)
            self.update_mean_posterior(arm_index, observed_reward)


class ThompsonGaussianKnownSigma(Strategy):
    def __init__(self, bandit, sigma, mu_prior, sigma_prior):
        self.bandit = bandit
        self.num_arms = len(bandit.get_arms_list())
        self.mu_posterior = [mu_prior] * self.num_arms
        self.sigma_posterior = [sigma_prior] * self.num_arms
        self.sigma = sigma

    def choose_arm(self):
        mean_samples = [np.random.normal(loc=mu, scale=sigma) for mu, sigma in
                        zip(self.mu_posterior, self.sigma_posterior)]
        return np.argmax(mean_samples)

    def pull_arm(self, arm_index):
        return self.bandit.pull_arm(arm_index)

    def update_mean_posterior(self, arm_index, observed_reward):
        temp_sigma = 1 / (1 / self.sigma_posterior[arm_index] ** 2 + 1 / self.sigma ** 2)
        self.mu_posterior[arm_index] = (self.mu_posterior[arm_index] / self.sigma_posterior[arm_index] ** 2 + observed_reward / self.sigma ** 2) * temp_sigma
        self.sigma_posterior[arm_index] = np.sqrt(temp_sigma)

    def fit(self, iterations):
        for iter in range(iterations):
            arm_index = self.choose_arm()
            observed_reward = self.pull_arm(arm_index)
            self.update_mean_posterior(arm_index, observed_reward)

class EpsilonGreedy(Strategy):
    """ Epislon Greedy Strategy
    
    """
    def __init__(self, bandit, epsilon, **kwargs):
        self.bandit = bandit
        self.epsilon = epsilon
        self.arms_list = bandit.get_arms_list()
#        self.estimate_arm_reward = np.zeros(self.arms_list.size)
        return

    def fit(self, iterations, **kwargs):
        """ Fit

        Args:
            iterations (int): number of iterations
            **kwargs: other

        Returns:
            A dictionary with arguments:
                rewards (list): the values returned by the bandit
                                at every iteration.
                arms_pulled (list): the arm pulled at every iteration.
                ...
        """
        assert(iterations >= len(self.arms_list))
        iteration = 0
        
        rewards_per_arm = [[] for i in range(len(self.arms_list))]
        rewards = [None] * iterations
        arms_pulled = [None] * iterations
                      
        estimated_arm_means = np.zeros((len(self.arms_list), iterations))
        

        def pull_arm_index(arm_index, iteration):
            arm = self.arms_list[arm_index]
            reward = self.bandit.pull_arm(arm)
            
            arms_pulled[iteration] = arm
            rewards[iteration] = reward
            rewards_per_arm[arm_index].append(reward)
            
            estimated_arm_means[:,iteration] = np.array(
                    [np.mean(a) for a in rewards_per_arm]
                    )
            return
            
        # Pull each arm once
        scan_order = np.arange(len(self.arms_list))
        np.random.shuffle(scan_order)
        for arm_index in scan_order:
            pull_arm_index(arm_index, iteration)
            iteration += 1
        
        # Epsilon Greedy
        while(iteration < iterations):
            if(np.random.rand() < self.epsilon):
                # Explore
                arm_index = np.random.randint(0, len(self.arms_list))
                pull_arm_index(arm_index, iteration)
                iteration += 1
                
            else:
                # Greedy
                arm_index = np.argmax([np.mean(a) for a in rewards_per_arm])
                pull_arm_index(arm_index, iteration)
                iteration += 1
                
            
        out_dict = dict(
                rewards = rewards,
                arms_pulled = arms_pulled,
                estimated_arm_means = estimated_arm_means,
                )
        return out_dict


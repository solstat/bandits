"""


"""

import numpy as np
import scipy.stats
from copy import deepcopy
import matplotlib.pyplot as plt

class Strategy:
    """ Base Strategy Class

    Args:
        bandit (Bandit): bandit to apply strategy on
        **kwargs: additonal key-word arguments

    """
    def __init__(self, bandit, **kwargs):
        raise NotImplementedError

    def fit(self, iterations, **kwargs):
        """ Fit Strategy on the current bandit

        Args:
            iterations (int): number of iterations to evaluate
            **kwargs: additional key-word arguments

        Returns:
            A dictionary with arguments:
                rewards (list): the values returned by the bandit
                                at every iteration.
                arms_pulled (list): the arm pulled at every iteration.
                ... : additional optional return values
        """
        raise NotImplementedError


class ThompsonSampling(Strategy):
    def __init__(self, bandit, **kwargs):
        self.bandit = bandit
        self.num_arms = bandit.num_arms
        self.prior_params = [deepcopy(kwargs) for _ in range(self.num_arms)]
        self.posterior_params = deepcopy(self.prior_params)

    def fit(self, iterations, restart=False, plot=True):
        if restart == True:
            self._restart()

        index_arms_pulled = [None] * iterations
        observed_rewards = [None] * iterations
        mean_reward_estimates = [None] * iterations
        for i in range(iterations):
            index_arms_pulled[i] = self._choose_arm()
            observed_rewards[i] = self._pull_arm(index_arms_pulled[i])
            self._update_posterior(index_arms_pulled[i], observed_rewards[i])
            mean_reward_estimates[i] = self.mean_reward_estimates

        if plot == True:
            plt.close()
            plt.plot(mean_reward_estimates)
            plt.show()

        out = {
            'rewards': observed_rewards,
            'arms_pulled': index_arms_pulled,
            'estimated_arm_means': mean_reward_estimates
        }
        return out

    def _restart(self):
        self.posterior_params = deepcopy(self.prior_params)

    def _choose_arm(self):
        samples = [self._sample(**params) for params in self.posterior_params]
        return np.argmax(samples)

    def _pull_arm(self, arm_index):
        return self.bandit.pull_arm(arm_index)

    def _sample(self, **kwargs):
        raise NotImplementedError

    def _update_posterior(self, arm_index, observed_reward):
        raise NotImplementedError

    @property
    def mean_reward_estimates(self):
        raise NotImplementedError


class ThompsonBernoulli(ThompsonSampling):
    """asdf"""

    def __init__(self, bandit, alpha_prior, beta_prior):
        ThompsonSampling.__init__(self, bandit, alpha=alpha_prior, beta=beta_prior)

    def _sample(self, alpha, beta):
        return scipy.stats.beta.rvs(alpha, beta, size=1)

    def _update_posterior(self, arm_index, observed_reward):
        if observed_reward == 1:
            self.posterior_params[arm_index]['alpha'] += 1
        else:
            self.posterior_params[arm_index]['beta'] += 1

    @property
    def mean_reward_estimates(self):
        return [params['alpha'] / (params['alpha'] + params['beta']) for params
                in self.posterior_params]


class ThompsonGaussianKnownSigma(ThompsonSampling):
    """asdf """

    def __init__(self, bandit, sigma, mu_prior, sigma_prior):
        ThompsonSampling.__init__(self, bandit, mu=mu_prior, sigma=sigma_prior)
        self.sigma = sigma

    def _sample(self, mu, sigma):
        return np.random.normal(loc=mu, scale=sigma)

    def _update_posterior(self, arm_index, observed_reward):
        sigma2 = self.sigma ** 2
        old_mu_posterior = self.posterior_params[arm_index]['mu']
        old_sigma2_posterior = self.posterior_params[arm_index]['sigma'] ** 2

        new_sigma_posterior = 1 / (1 / old_sigma2_posterior + 1 / sigma2)
        new_mu_posterior = (old_mu_posterior / old_sigma2_posterior + observed_reward / sigma2) * new_sigma_posterior
        self.posterior_params[arm_index]['mu'] = new_mu_posterior
        self.posterior_params[arm_index]['sigma'] = np.sqrt(new_sigma_posterior)

    @property
    def mean_reward_estimates(self):
        return [params['mu'] for params in self.posterior_params]


class EpsilonGreedy(Strategy):
    """ Epislon Greedy Strategy

    Args:
        bandit (Bandit): bandit of interest
        epsilon (double): chance of random exploration vs exploitation
            must be a value within [0,1]

    """
    def __init__(self, bandit, epsilon, **kwargs):
        self.bandit = bandit
        if epsilon > 1 or epsilon < 0:
            raise ValueError("epsilon {0} must be in [0,1]".format(epsilon))
        self.epsilon = epsilon
        self.num_arms = self.bandit.num_arms
        return

    def fit(self, iterations, memory_multiplier = 1.0, **kwargs):
        """ Fit

        Args:
            iterations (int): number of iterations
            memory_multiplier (double): exponential decay factor for arm
                estimates. Must be in (0,1]
            **kwargs: other

        Returns:
            A dictionary with arguments:
                rewards (list): the values returned by the bandit
                                at every iteration.
                arms_pulled (list): the arm pulled at every iteration.
                ...
        """
        assert(iterations >= self.num_arms)
        assert(memory_multiplier > 0)
        assert(memory_multiplier <= 1)

        iteration = 0

        reward_sum_per_arm = np.zeros(self.num_arms)
        reward_count_per_arm = np.zeros(self.num_arms)
        estimated_arm_means = np.zeros((self.num_arms, iterations))

        rewards = [None] * iterations
        arms_pulled = [None] * iterations


        def pull_arm_index(arm_index, iteration):
            nonlocal rewards, arms_pulled, \
                reward_sum_per_arm, reward_count_per_arm, estimated_arm_means
            reward = self.bandit._pull_arm(arm_index)

            # Update statistics
            arms_pulled[iteration] = arm_index
            rewards[iteration] = reward
            reward_sum_per_arm[arm_index] += reward
            reward_count_per_arm[arm_index] += 1

            # Calculate estimate of arm means
            nonzero_arms = reward_count_per_arm > 0
            estimated_arm_means[nonzero_arms, iteration] = (
                    reward_sum_per_arm[nonzero_arms] /
                    reward_count_per_arm[nonzero_arms]
                    )
            estimated_arm_means[~nonzero_arms, iteration] = np.nan

            # Apply memory_multiplier
            reward_sum_per_arm *= memory_multiplier
            reward_count_per_arm *= memory_multiplier
            return

        # Pull each arm once
        scan_order = np.arange(self.num_arms)
        np.random.shuffle(scan_order)
        for arm_index in scan_order:
            pull_arm_index(arm_index, iteration)
            iteration += 1

        # Epsilon Greedy
        while(iteration < iterations):
            if(np.random.rand() < self.epsilon):
                # Explore
                arm_index = np.random.randint(0, self.num_arms)
                pull_arm_index(arm_index, iteration)
                iteration += 1

            else:
                # Greedy
                arm_index = np.argmax(estimated_arm_means[:,iteration-1])
                pull_arm_index(arm_index, iteration)
                iteration += 1


        out_dict = dict(
                rewards = rewards,
                arms_pulled = arms_pulled,
                estimated_arm_means = estimated_arm_means,
                )
        return out_dict


class UCB(Strategy):
    """ UCB Strategy

    Args:
        bandit (Bandit): bandit of interest
        epsilon (double): chance of random exploration vs exploitation
            must be a value within [0,1]

    """
    def __init__(self, bandit, **kwargs):
        self.bandit = bandit
        self.num_arms = self.bandit.num_arms
        return

    def fit(self, iterations, **kwargs):
        """ Fit

        Args:
            iterations (int): number of iterations

        Returns:
            A dictionary with arguments:
                rewards (list): the values returned by the bandit
                                at every iteration.
                arms_pulled (list): the arm pulled at every iteration.
                ...
        """
        assert(iterations >= self.num_arms)

        iteration = 0

        reward_sum_per_arm = np.zeros(self.num_arms)
        reward_count_per_arm = np.zeros(self.num_arms)
        estimated_arm_means = np.zeros((self.num_arms, iterations))

        rewards = [None] * iterations
        arms_pulled = [None] * iterations

        def pull_arm_with_index(arm_index, iteration):
            nonlocal rewards, arms_pulled, \
                reward_sum_per_arm, reward_count_per_arm, estimated_arm_means
            reward = self.bandit._pull_arm(arm_index)
            if reward < 0 or reward > 1:
                raise Exception("UCB bandit algorithm only works when bandit arms " +
                                "return rewards between 0 and 1.")

            # Update statistics
            arms_pulled[iteration] = arm_index
            rewards[iteration] = reward
            reward_sum_per_arm[arm_index] += reward
            reward_count_per_arm[arm_index] += 1

            # Calculate estimate of arm means
            nonzero_arms = reward_count_per_arm > 0
            estimated_arm_means[nonzero_arms, iteration] = (
                    reward_sum_per_arm[nonzero_arms] /
                    reward_count_per_arm[nonzero_arms]
                    )
            estimated_arm_means[~nonzero_arms, iteration] = np.nan

            return

        # Pull each arm once
        scan_order = np.arange(self.num_arms)
        np.random.shuffle(scan_order)
        for arm_index in scan_order:
            pull_arm_with_index(arm_index, iteration)
            iteration += 1

        # UCB alg
        while(iteration < iterations):
            arm_index = np.argmax(estimated_arm_means[:,iteration - 1] + np.sqrt(2 * np.log(iterations) / reward_count_per_arm))

            pull_arm_with_index(arm_index, iteration)
            iteration += 1

        out_dict = dict(
                rewards = rewards,
                arms_pulled = arms_pulled,
                estimated_arm_means = estimated_arm_means,
                )
        return out_dict


import numpy as np
from scipy.stats import beta

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


class ThompsonBernoulli(Strategy):
    def __init__(self, bandit, alpha_prior, beta_prior):
        self.bandit = bandit
        self.num_arms = bandit.num_arms
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
        self.num_arms = bandit.num_arms
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
            reward = self.bandit.pull_arm(arm_index)

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


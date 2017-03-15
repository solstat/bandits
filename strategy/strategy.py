"""

Base Strategy Class

"""

import numpy as np
import scipy.stats
from copy import deepcopy

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
        pull_count_per_arm = np.zeros(self.num_arms)
        estimated_arm_means = np.zeros((self.num_arms, iterations))

        rewards = [None] * iterations
        arms_pulled = [None] * iterations


        def pull_arm_index(arm_index, iteration):
            nonlocal rewards, arms_pulled, \
                reward_sum_per_arm, pull_count_per_arm, estimated_arm_means
            reward = self.bandit.pull_arm(arm_index)

            # Update statistics
            arms_pulled[iteration] = arm_index
            rewards[iteration] = reward
            reward_sum_per_arm[arm_index] += reward
            pull_count_per_arm[arm_index] += 1

            # Calculate estimate of arm means
            nonzero_arms = pull_count_per_arm > 0
            estimated_arm_means[nonzero_arms, iteration] = (
                    reward_sum_per_arm[nonzero_arms] /
                    pull_count_per_arm[nonzero_arms]
                    )
            estimated_arm_means[~nonzero_arms, iteration] = np.nan

            # Apply memory_multiplier
            reward_sum_per_arm *= memory_multiplier
            pull_count_per_arm *= memory_multiplier
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
                arm_index = int(np.random.randint(0, self.num_arms))
                pull_arm_index(arm_index, iteration)
                iteration += 1

            else:
                # Greedy
                arm_index = int(np.argmax(estimated_arm_means[:,iteration-1]))
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
        pull_count_per_arm = np.zeros(self.num_arms)
        estimated_arm_means = np.zeros((self.num_arms, iterations))

        rewards = [None] * iterations
        arms_pulled = [None] * iterations

        def pull_arm_with_index(arm_index, iteration):
            nonlocal rewards, arms_pulled, \
                reward_sum_per_arm, pull_count_per_arm, estimated_arm_means
            reward = self.bandit.pull_arm(arm_index)
            if reward < 0 or reward > 1:
                raise Exception("UCB bandit algorithm only works when bandit arms " +
                                "return rewards between 0 and 1.")

            # Update statistics
            arms_pulled[iteration] = arm_index
            rewards[iteration] = reward
            reward_sum_per_arm[arm_index] += reward
            pull_count_per_arm[arm_index] += 1

            # Calculate estimate of arm means
            nonzero_arms = pull_count_per_arm > 0
            estimated_arm_means[nonzero_arms, iteration] = (
                    reward_sum_per_arm[nonzero_arms] /
                    pull_count_per_arm[nonzero_arms]
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
            arm_index = np.argmax(estimated_arm_means[:,iteration - 1] +
                    np.sqrt(2 * np.log(iterations) / pull_count_per_arm))

            pull_arm_with_index(arm_index, iteration)
            iteration += 1

        out_dict = dict(
                rewards = rewards,
                arms_pulled = arms_pulled,
                estimated_arm_means = estimated_arm_means,
                )
        return out_dict


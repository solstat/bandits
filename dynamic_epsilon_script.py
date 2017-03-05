#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:05:20 2017

@author: nghoiyi
"""

import matplotlib.pyplot as plt
import numpy as np
from bandit import LinearInterpolationBandit
from strategy import EpsilonGreedy, UCB
import matplotlib.pyplot as plt

# Main Script
if __name__ == "__main__":
    print("practice script py")
    my_bandit = LinearInterpolationBandit(means=np.array([[.5], [.7]]),
                                          periods = [200],
                                          noise_func=lambda x: np.random.binomial(1, x))
    #my_strategy = EpsilonGreedy(bandit = my_bandit, epsilon = 0.1)
    #out = my_strategy.fit(iterations=1000, memory_multiplier = 0.9)
    my_strategy = UCB(bandit = my_bandit)
    out = my_strategy.fit(iterations=1000)
    print("Average Reward: " + str(np.mean(out['rewards'])))
    plt.close()
    plt.plot(out['estimated_arm_means'][:, my_bandit.num_arms:].T)
    plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:05:20 2017

@author: nghoiyi
"""

import numpy as np
from bandit import DynamicBandit
from strategy import EpsilonGreedy
import matplotlib.pyplot as plt


# Main Script
if __name__ == "__main__":
    print("practice script py")
    my_bandit = DynamicBandit(means=np.array([[10, 0],[0,10]]),
                              periods = [200, 200])
    my_strategy = EpsilonGreedy(bandit = my_bandit, epsilon = 0.1)
    out = my_strategy.fit(iterations=1000, memory_multiplier = 0.9)
    print("Average Reward: " + str(np.mean(out['rewards'])))
    plt.close()
    plt.plot(out['estimated_arm_means'].T)
    plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:05:20 2017

@author: nghoiyi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sst_bandit import SSTBandit
from strategy import EpsilonGreedy

# Main Script
if __name__ == "__main__":
    sst_df = pd.read_csv("./data/coarse_sst.csv")
    my_bandit = SSTBandit(sst_df=sst_df, pulls_per_month=10)
    my_strategy = EpsilonGreedy(bandit=my_bandit, epsilon=0.1)
    out = my_strategy.fit(iterations=10000)

    # print("Average Reward: " + str(np.mean(out['rewards'])))
    # plt.close()
    # plt.plot(out['estimated_arm_means'][:, my_bandit.num_arms:].T)
    # plt.show()

    x = out['estimated_arm_means'][:,-1]

    x_mat = np.zeros((36, 18)) * np.nan

    for lon, lat, x_i in zip(my_bandit.lon, my_bandit.lat, x):
        x_mat[np.floor(lon / 10), np.floor((90 + lat) / 10)] = x_i

    plt.close()
    plt.imshow(x_mat.T[-1:0:-1,:])
    plt.show()




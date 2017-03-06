import pandas as pd
import numpy as np

def run_strategies(strategy_builders, bandit_builder,
                   iterations, replications=1):

    all_rewards = []
    all_cum_rewards = []
    all_arms_pulled = []
    all_names = []
    all_iterations = []
    all_replications = []

    for i in range(len(strategy_builders)):
        name = strategy_builders[i][1]
        for j in range(replications):
            bandit = bandit_builder()
            strategy = strategy_builders[i][0](bandit)
            fit_params = strategy_builders[i][2]

            strategy_result = strategy.fit(iterations, **fit_params)

            all_rewards.extend(strategy_result["rewards"])
            all_cum_rewards.extend(np.cumsum(strategy_result["rewards"]))
            all_arms_pulled.extend(strategy_result["arms_pulled"])
            all_names.extend([name] * iterations)
            all_iterations.extend(range(iterations))
            all_replications.extend([j] * iterations)

    return pd.DataFrame({"reward" : all_rewards, "cum_reward" : all_cum_rewards,
                         "arm_pulled" : all_arms_pulled,
                         "strategy_name" : all_names, "iteration" : all_iterations,
                         "replication" : all_replications})

# def plot_average_rewards(strategy_rewards_df):
#     import matplotlib.pyplot as plt
#     strategy_rewards_df = strategy_rewards_df.drop(["replication", "arm_pulled"], axis=1)
#     means = strategy_rewards_df.groupby(["strategy_name", "iteration"]).mean().reset_index()

import bandit as bt
import strategy as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Main Script
if __name__ == "__main__":
    print("practice script py")
    bandit_builder = lambda: bt.LinearInterpolationBandit(means=np.array([[0.8,0.8,0.2,0.2], [0.2,0.2,0.8,0.8]]),
                                      periods = [40,10,40,10],
                                      noise_func=lambda x: np.random.binomial(1, x))
    #my_strategy = EpsilonGreedy(bandit = my_bandit, epsilon = 0.1)
    #out = my_strategy.fit(iterations=1000, memory_multiplier = 0.9)
    strategy_builders = [(lambda x: st.UCB(bandit = x), "UCB", {}),
                         (lambda x: st.EpsilonGreedy(x, .1), "Ep Greedy", {}),
                         (lambda x: st.EpsilonGreedy(x, .01), "Ep Greedy (Mem Mult 0.9)", {"memory_multiplier": .5})]
    out = run_strategies(strategy_builders, bandit_builder, 300, 100)

    plt.close()
    sns.tsplot(time="iteration", value="cum_reward", condition="strategy_name", unit="replication", data=out, ci=100)
    plt.show()


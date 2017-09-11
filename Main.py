""" Solving the CartPole-v0 environment with the Cross Entropy Method """

__author__ = "Jan Krepl"
__email__ = "jankrepl@yahoo.com"
__license__ = "MIT"

import gym
from foo import *

# PARAMETERS
init_mu = [0, 0, 0, 0]
init_covmat = 1 * np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
episodes_for_evaluation = 20  # each policy run for episodes_for_evaluation episodes and then result is the mean score
n_samples = 100  # number of sampled weights we generate
n_best_to_keep = 5  # We order policies based on results and then use n_best_to_keep best of them to estimate new param
absolute_winners_threshold = 200  # algorithm terminated when a certain number of winning policies found
initial_noise_coef = 1  # we always add constant_noise_coef * I to the estimated covmat (to increase variance)
noise_decay = 99 / 100

# INITIALIZATION
env = gym.make('CartPole-v0')

global_max = 0
absolute_winners = []  # list of policies(w) that achieved perfect score of 199
mu = init_mu
covmat = init_covmat
noise_coef = initial_noise_coef

counter = 0

# MAIN ALGORITHM
while len(absolute_winners) < absolute_winners_threshold:

    # 1) Sample policies
    samples = simulator(n_samples, mu, covmat)

    # 2) Evaluate policies and append absolute winners
    samples_result = {w: evaulate_policy(w, env, number_of_episodes=episodes_for_evaluation) for w in samples}

    for w, res in samples_result.items():
        if res == 200:
            absolute_winners.append(w)

    # 3) Pick the n_best_to_keep best policies
    local_winners = sorted(samples_result.keys(), key=lambda my_key: samples_result[my_key])[-n_best_to_keep:]

    # 4) Estimate new parameters and decay noise - in case we managed to find a better solution
    current_max = np.mean(sorted(samples_result.values())[-n_best_to_keep:])
    print('Current max is: ' + str(current_max))
    if global_max < current_max:
        global_max = current_max
        mu, covmat = estimator(local_winners, noise_coef)
        noise_coef *= noise_decay

    # 5) Counter and results
    counter += 1
    print('The algorithm finished its ' + str(counter) + ' iteration')
    print('So far, it has found ' + str(len(absolute_winners)) + ' winners')
    print('The noise_coef: ' + str(noise_coef))

env.close()

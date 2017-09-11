import numpy as np


def evaulate_policy(w, env, number_of_episodes=2):
    """It evaluates a policy for number_of_episodes and returns and average score

    :param w: our policy is inner(w, s) > 0
    :type w: ndarray
    :param number_of_episodes: number of episodes we will run the policy for
    :type number_of_episodes: int
    :param env: environment object
    :type env: environment object
    :return: sum(timesteps_i)/number_of_episodes
    :rtype: float
    """

    results = []
    for e in range(number_of_episodes):
        s_old = env.reset()
        t = 0
        done = False
        while not done:
            # Choose action
            action = None
            if np.inner(w, s_old) > 0:
                action = 1
            else:
                action = 0
            # Take action
            s_new, r, done, _ = env.step(action)

            # Update
            s_old = s_new

            t += 1
        results.append(t)

    return np.mean(results)


def estimator(w_list, noise_coef):
    """ It estiamted the mean vector and covariance matrix based on the list of collected w

    :param w_list: list of weights that we will use to form our estimates
    :type w_list: list of tuples
    :param noise_coef: to estimated covariance matrix we add a noise_coef * identity matrix to increase variance
    :type noise_coef: float
    :return: sample estimate of mean vector (4,) and covariance matrix (4,4)
    :rtype: pair of ndarrays
    """
    w_list_ndarray = np.array(w_list)
    mu_hat = np.mean(w_list_ndarray, axis=0)
    covmat_hat = np.cov(np.transpose(w_list_ndarray)) + noise_coef * np.eye(4, 4)  # ADD SOME CONSTANT TO AVOID
    return mu_hat, covmat_hat


def simulator(n, mu, covmat):
    """ Sampling n samples from multivariate normal with mean vector mu and covariance matrix covmat

    :param n: number of samples to generate
    :type n: int
    :param mu: mean vector of 4 elements
    :type mu: ndarray
    :param covmat: (4,4) ndarray - covariance matrix
    :type covmat: ndarray
    :return: samples of multivariate normal
    :rtype: list of tuples
    """
    a = np.random.multivariate_normal(mu, covmat, n)
    return [tuple(a[i, :]) for i in range(n)]

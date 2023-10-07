import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from warnings import catch_warnings, simplefilter
from pdb import set_trace

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.v_min = 1.2
        self.unsafe_prob = .05

        f_kernel = .5 * Matern(length_scale=.5, nu=.5)
        self.f_model = GaussianProcessRegressor(kernel=f_kernel, random_state=0)

        v_kernel = 1.5 + np.sqrt(2) * Matern(length_scale=.5, nu=2.5)
        self.v_model = GaussianProcessRegressor(kernel=v_kernel, random_state=0)

        self.X = np.array([])
        self.f = np.array([])
        self.v = np.array([])

        self.f_hat_opt = None

        self.beta = 6.0
        self.beta_discount = 0.95

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        return self.optimize_acquisition_function()


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        f_mu, f_std = self.predict_f(np.atleast_2d(x))
        v_mu, v_std = self.predict_v(np.atleast_2d(x))
        f_mu, f_std, v_mu, v_std = f_mu[0], f_std[0], v_mu[0], v_std[0]

        # Method
        method = 3
        if method == 0:
            # Pick random points
            return 0.
        elif method == 1:
            # Probability of Improvement
            if self.f_hat_opt is not None:
                prob = norm.cdf((f_mu-self.f_hat_opt)/(f_std+1e-9))
                prob = prob * (1 - norm.cdf((self.v_min-v_mu)/v_std))
                return prob
            else:
                return 0.
        elif method == 2:
            # Expected improvement
            if f_std == 0. or self.f_hat_opt is None:
                return 0.
            else:
                z = (f_mu-self.f_hat_opt-0.01)/f_std
                ei = (f_mu-self.f_hat_opt)*(norm.cdf(z))+(f_std)*(norm.pdf(z))
                ei = ei * (1 - norm.cdf((self.v_min-v_mu)/v_std))
                return ei
        elif method == 3:
            # UCB
            ucb = 3.0 * (f_mu + self.beta * f_std) + 1.0 * (v_mu + self.beta * v_std)
            return ucb

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        self.X = np.append(self.X, np.atleast_1d(x)[0]).reshape(-1,1)
        self.f = np.append(self.f, np.atleast_1d(f)[0])
        self.v = np.append(self.v, np.atleast_1d(v)[0])

        with catch_warnings():
            simplefilter("ignore")
            self.f_model.fit(self.X, self.f)
            self.v_model.fit(self.X, self.v)

        v_mu, v_std = self.predict_v(self.X)
        self.safes = self.compute_safe_indices(v_mu, v_std, unsafe_prob=self.unsafe_prob)

        f_mu, f_std = self.predict_f(self.X)
        self.f_hat = f_mu

        f_safes = self.f_hat[self.safes]
        self.f_hat_opt = np.max(f_safes) if len(f_safes) > 0 else self.f_hat_opt

        self.beta = self.beta_discount * self.beta # for UCB

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        X = np.arange(0, 5, 0.001)
        f_mu, f_std = self.predict_f(X.reshape(-1,1))
        v_mu, v_std = self.predict_v(X.reshape(-1,1))

        safes = self.compute_safe_indices(v_mu, v_std, unsafe_prob=self.unsafe_prob)

        if len(safes)==0:
            return np.atleast_2d(np.random.choice(X))

        X_safe = X[safes]
        f_safe = f_mu[safes]

        return np.atleast_2d(X_safe[np.argmax(f_safe)])

    def predict_f(self, X):
        with catch_warnings():
            simplefilter("ignore")
            f_mu, f_std = self.f_model.predict(X, return_std=True)
        return f_mu, f_std

    def predict_v(self, X):
        with catch_warnings():
            simplefilter("ignore")
            v_mu, v_std = self.v_model.predict(X, return_std=True)
        return v_mu, v_std

    def is_safe(self, v_mu, v_std, unsafe_prob):
        return norm.cdf((self.v_min-v_mu)/(v_std+1e-9)) <= unsafe_prob

    def compute_safe_indices(self, v_mu, v_std, unsafe_prob):
        assert len(v_mu) > 0, "len(v_mu) should be larger than 0."
        safes = np.array([i for i, (mu, std) in enumerate(zip(v_mu, v_std)) if self.is_safe(mu, std, unsafe_prob)], dtype=int)
        return safes


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()
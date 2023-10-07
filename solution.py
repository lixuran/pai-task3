def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel
from scipy.stats import norm
#from sklearn.utils.testing import ignore_warnings
#from sklearn.exceptions import ConvergenceWarning
domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

""" Solution """

#@ignore_warnings(category=ConvergenceWarning)
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        sigma_f = 0.15
        sigma_v = 0.0001
        self.xs  = np.array([[]])
        self.vs  = np.array([[]])
        self.fs  = np.array([[]])
        self.kernel_f = 0.5 * Matern(length_scale=0.5,length_scale_bounds="fixed", nu=2.5) +WhiteKernel(sigma_f,noise_level_bounds="fixed")            #todo: check change value here
        self.kernel_v = 1.5+np.sqrt(2) * Matern(length_scale=0.5,length_scale_bounds="fixed", nu=2.5)+WhiteKernel(sigma_v,noise_level_bounds="fixed")  
        self.gpr_f = GaussianProcessRegressor(kernel=self.kernel_f,
         random_state=SEED)
        
        self.gpr_v = GaussianProcessRegressor(kernel=self.kernel_v,
         random_state=SEED)
        self.k=6
        self.discount = 0.95
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
        return self.optimize_acquisition_function() #todo: maybe only sample in the interval constrained by v?


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
       
        #return 0
        method = 2
        mean,std = self.gpr_f.predict(x.reshape(-1,1),return_std=True) #todo: calculate af_value
        if method ==1:
        #using EI
            f_best = np.max(self.fs)
            #print("fbest shape",f_best.shape)
            rho = (mean - f_best)/std
            
            af_value = std*(rho*norm.cdf(rho)+norm.pdf(rho))
        #using UCB what k to use?
        #
        elif method ==2:
            af_value = mean+self.k*std
        mean_v,std_v = self.gpr_v.predict(x.reshape(-1,1),return_std=True)
        #norm_threshold = (SAFETY_THRESHOLD- mean_v)/std_v
        #af_value = af_value*(1-norm.cdf(norm_threshold))
        af_value = af_value+(mean_v+self.k*std_v)*0.3
        #print("af_value shape",af_value.shape)
        return af_value.flatten()
        #raise NotImplementedError


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
        self.xs = np.append(self.xs,x)
        self.fs = np.append(self.fs,f)
        self.vs = np.append(self.vs,v)
        #print(self.xs)
        #print(self.xs.reshape(-1,1))
        self.gpr_f.fit(self.xs.reshape(-1,1), self.fs.reshape(-1,1))
        self.gpr_v.fit(self.xs.reshape(-1,1), self.vs.reshape(-1,1))
        self.k = self.k*self.discount

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter constraints for where to return for the f values: i.e. only on places where v is most likely higher than threshold.
        # def objective(x):
        #     return -self.gpr_f.predict(x.reshape(-1,1)).flatten()

        # f_values = []
        # x_values = []
        
        # # Restarts the optimization 20 times and pick best solution
        # for _ in range(20):
        #     x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
        #          np.random.rand(domain.shape[0])
        #     result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
        #                            approx_grad=True)
        #     x_values.append(np.clip(result[0], *domain[0]))
        #     f_values.append(-result[1])

        # ind = np.argmax(f_values)
        #print("domain shape",np.atleast_2d(x_values[ind]))
        
        # xs  = np.array([5/5000*i for i in range(5000)])
        # max_f =-1
        # max_loc = -1
        # max_f_worst=-1
        # max_loc_worst= -1
        # for i in xs:
        #     mean_v,std_v = self.gpr_v.predict(i.reshape(-1,1),return_std=True)
        #     norm_threshold = (SAFETY_THRESHOLD- mean_v)/std_v
            
        #     mean= self.gpr_f.predict(i.reshape(-1,1))
        #     if(mean>max_f):
        #         if(norm.cdf(norm_threshold)<0.05):
        #             max_f = mean
        #             max_loc = i
        #         if(max_f_worst<mean):
        #             max_f_worst = mean
        #             max_loc_worst = i
        # if max_f ==-1:
        #     max_loc = max_loc_worst
        # return np.atleast_2d(max_loc)
        
        safes = self.compute_safe_indices(v_mu, v_std, unsafe_prob=self.unsafe_prob)

        if len(safes)==0:
            return np.atleast_2d(np.random.choice(X))

        X_safe = X[safes]
        f_safe = f_mu[safes]

        return np.atleast_2d(X_safe[np.argmax(f_safe)])


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
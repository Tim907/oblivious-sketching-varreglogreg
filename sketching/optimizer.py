import numpy as np
import scipy.optimize as so
from numba import jit
from sklearn.linear_model import SGDClassifier


def only_keep_k(vec, block_size, k, max_len=None, biggest=True):
    """
    Only keep the k biggest (smalles) elements for each block in a vector.

    If max_len = None, use the whole vec. Otherwise, use vec[:max_len]

    Returns: new vector, indices
    """

    if k == block_size:
        return vec, np.array(list(range(len(vec))))

    do_not_touch = np.array([])
    if max_len is not None:
        do_not_touch = vec[max_len:]
        vec = vec[:max_len]

    # determine the number of blocks
    num_blocks = int(vec.shape[0] / block_size)

    # split the vector in a list of blocks (chunks)
    chunks = np.array_split(vec, num_blocks)

    # chunks_new will contain the k biggest (smallest) elements for each chunk
    chunks_new = []
    keep_indices = []
    for i, cur_chunk in enumerate(chunks):
        if biggest:
            cur_partition_indices = np.argpartition(-cur_chunk, k)
        else:
            cur_partition_indices = np.argpartition(cur_chunk, k)
        chunks_new.append(cur_chunk[cur_partition_indices[:k]])
        keep_indices.extend(cur_partition_indices[:k] + i * block_size)

    if max_len is not None:
        chunks_new.append(do_not_touch)
        keep_indices.extend(
            list(range(vec.shape[0], vec.shape[0] + do_not_touch.shape[0]))
        )

    return np.concatenate(chunks_new), np.array(keep_indices)


@jit(nopython=True)
def calc(v):
    """calculates log(1 + exp(v)), but becomes identity function if v is bigger than 34"""
    if v < 34:
        # prevent underflow exception
        if(v < -200): 
            return np.exp(-200)

        return np.log1p(np.exp(v))
    return v
        

calc_vectorized = np.vectorize(calc)


def logistic_likelihood(theta, Z, weights=None, block_size=None, k=None, max_len=None):
    v = -Z.dot(theta)
    if block_size is not None and k is not None:
        v, indices = only_keep_k(v, block_size, k, max_len=max_len, biggest=True)
        if weights is not None:
            weights = weights[indices]
    likelihoods = calc_vectorized(v)
    if weights is not None:
        likelihoods = weights * likelihoods.T
    return np.sum(likelihoods)


def logistic_likelihood_varregul(theta, Z, weights=None, block_size=None, k=None, max_len=None, lamb=0):
    """the variance regularized version of logistic_likelihood"""

    v = -Z.dot(theta)
    if block_size is not None and k is not None:
        v, indices = only_keep_k(v, block_size, k, max_len=max_len, biggest=True)
        if weights is not None:
            weights = weights[indices]
    likelihoods = calc_vectorized(v)
    if weights is not None:
        term1 = np.sum(weights * likelihoods.T)
        term2 = lamb / 2 * np.sum(weights * np.square(likelihoods).T)
    else:
        term1 = np.sum(likelihoods)
        term2 = lamb / 2 * np.sum(np.square(likelihoods).T)
        
    # take the sum of the three summands
    return sum((
        term1,
        term2,
        -lamb / (2 * Z.shape[0]) * term1 ** 2
    ))


def logistic_likelihood_grad(
    theta, Z, weights=None, block_size=None, k=None, max_len=None):
    v = Z.dot(theta)
    if block_size is not None and k is not None:
        v, indices = only_keep_k(v, block_size, k, max_len=max_len, biggest=False)
        if weights is not None:
            weights = weights[indices]
        Z = Z[indices, :]

    grad_weights = 1.0 / (1.0 + np.exp(v))

    if weights is not None:
        grad_weights *= weights

    return -1 * (grad_weights.dot(Z))


def logistic_likelihood_grad_varregul(
    theta, Z, weights=None, block_size=None, k=None, max_len=None, lamb=0):
    """the variance regularized version of the logistic likelihood gradient"""

    v = Z.dot(theta)
    if block_size is not None and k is not None:
        v, indices = only_keep_k(v, block_size, k, max_len=max_len, biggest=False)
        if weights is not None:
            weights = weights[indices]
        Z = Z[indices, :]

    sigmoid_term = 1.0 / (1.0 + np.exp(np.minimum(34, np.maximum(-34, v))))
    log_term = calc_vectorized(-v)

    grad_weights_1 = sigmoid_term
    weight_log_term = log_term
    if weights is not None:
        grad_weights_1 *= weights
        weight_log_term *= weights
    
    grad_weights_2 = weight_log_term * sigmoid_term
    final_term_1 = grad_weights_1.dot(-Z)

    # take the sum of the three summands
    return sum((final_term_1,
        lamb * grad_weights_2.dot(-Z),
        -lamb / Z.shape[0] * np.sum(weight_log_term) * final_term_1
    ))


def L1_objective(theta, X, y):
    """L1 loss function used in cauchy-sketching"""
    return np.sum(np.abs(X.dot(theta) - y))

def L1_grad(theta, X, y):
    """L1 gradient function used in cauchy-sketching"""
    return sum(np.multiply(X, np.sign(X.dot(theta) - y)[:, np.newaxis]))


def optimize(Z, w=None, block_size=None, k=None, max_len=None, varreg_lambda=0):
    """Optimizes a weighted instance of logistic regression with given hyperparameter lambda of variance-regularization."""

    if w is None:
        w = np.ones(Z.shape[0])

    def objective_function(theta):

        if(varreg_lambda == 0):
            return logistic_likelihood(theta, Z, w, block_size=block_size, k=k, max_len=max_len)

        return logistic_likelihood_varregul(
            theta, Z, w, block_size=block_size, k=k, max_len=max_len, lamb=varreg_lambda
        )

    def gradient(theta):

        if(varreg_lambda == 0):
            return logistic_likelihood_grad(theta, Z, w, block_size=block_size, k=k, max_len=max_len)
            
        return logistic_likelihood_grad_varregul(
            theta, Z, w, block_size=block_size, k=k, max_len=max_len, lamb=varreg_lambda
        )

    theta0 = np.zeros(Z.shape[1])

    return so.minimize(objective_function, theta0, method="L-BFGS-B", jac=gradient)


def optimize_L1(Z):
    """Optimizes by L1 loss according to Theorem 1 of the paper."""

    X = Z[:, 0:(Z.shape[1]-1)]
    y = Z[:, -1]

    def objective_function(theta):
        return L1_objective(theta, X, y)

    def gradient(theta):
        return L1_grad(theta, X, y)

    theta0 = np.zeros(X.shape[1])

    return so.minimize(objective_function, theta0, method="L-BFGS-B", jac=gradient)


class base_optimizer:
    """optimizer for logistic regression"""

    def __init__(self) -> None:
        self.varreg_lambda = 0

    def setDataset(self, X, y, Z):
        self.X = X
        self.y = y
        self.Z = Z

    def get_Z(self):
        return self.Z

    def optimize(self, reduced_matrix, weights=None):
        return optimize(Z = reduced_matrix, w = weights).x

    def get_objective_function(self):
        return lambda theta: logistic_likelihood(theta, self.Z)


class varreg_optimizer(base_optimizer):
    """optimizer for variance-regularized logistic regression"""
    def __init__(self, varreg_lambda):
        super().__init__()
        self.varreg_lambda = varreg_lambda

    def optimize(self, reduced_matrix, weights=None):
        return optimize(Z = reduced_matrix, w = weights, varreg_lambda=self.varreg_lambda).x

    def get_objective_function(self):
        return lambda theta: logistic_likelihood_varregul(theta, self.Z, lamb=self.varreg_lambda)


class L1_optimizer(base_optimizer):
    """optimizer for L1 optimization used in cauchy-sketch"""
    def optimize(self, reduced_matrix, weights=None):
        return optimize_L1(reduced_matrix).x

    def get_objective_function(self):
        Z = self.get_Z()
        return lambda theta: L1_objective(theta, X=Z[:, 0:(Z.shape[1]-1)], y=Z[:, -1])

    def get_Z(self):
        return np.append(np.append(self.X, np.ones(shape=(self.X.shape[0], 1)), axis=1), self.y[:, np.newaxis], axis=1)


class sgd_optimizer(varreg_optimizer):
    """optimizer for stochastic gradient descent"""
    def optimize(self, reduced_matrix, weights=None):
        """Performs a run of stochastic gradient descent."""
        X = self.X
        y = self.y

        learner = SGDClassifier(
            loss="log", alpha=0, learning_rate="adaptive", eta0=0.01, max_iter=1
        )

        learner.partial_fit(X, y, classes=np.unique(y))

        beta_opt = np.concatenate((learner.coef_[0], learner.intercept_))

        return beta_opt
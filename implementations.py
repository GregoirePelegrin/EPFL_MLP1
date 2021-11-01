import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.zeros([x.shape[0], degree+1])
    for i in range(phi.shape[0]):
        for deg in range(phi.shape[1]):
            phi[i, deg] = x[i] ** deg
    return phi

def sigmoid(t):
    """apply the sigmoid function on t."""
    res = 1/(1+np.exp(-t))
    return res

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio"""
    np.random.seed(seed)
    indices = np.random.permutation(x.shape[0])
    if ratio == 0:
        separator_idx = int(0.8*x.shape[0])
    else:
        separator_idx = int(ratio*x.shape[0])
    training_idx, test_idx = indices[:separator_idx], indices[separator_idx:]
    return [x[training_idx], y[training_idx]], [x[test_idx], y[test_idx]]

def plot_train_test(train_errors, test_errors):
    ax = plt.subplot()
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 2*len(train_errors))))
    temp_x = np.arange(len(train_errors))
    ax.plot(temp_x, train_errors, label="tr", c=next(color))
    ax.plot(temp_x, test_errors, label="te", c=next(color))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.legend()
    plt.grid(b=True, which='major', linewidth=1.5)
    plt.grid(b=True, which='minor', linestyle="-.", linewidth=1)
    plt.plot

def standardize(x: np.matrix) -> np.matrix:
    ''' standardize the data by statistical analysis using np operations'''
    return (x - np.min(x))/np.ptp(x)


def compute_loss_MSE(y, tx, w):
    """Compute the loss with square error"""
    e = y - tx@w
    loss = 1/(2*y.shape[0]) * e.T@e
    return loss

def compute_loss_RMSE(y, tx, w):
    loss = (2*compute_loss_MSE(y, tx, w))
    return loss[0, 0]**0.5

def compute_loss_sigmoid(y, tx, w):
    """compute the loss: negative log likelihood."""
    loss = 0
    for xi,yi in zip(tx, y):
        temp = sigmoid(xi.T@w)[0, 0]
        print(temp)
        loss += yi*np.log(temp) + (1-yi)*np.log(1-temp)
    return -loss

def compute_gradient_MSE(y, tx, w):
    """Compute the gradient."""
    e = y - tx@w
    gradient = -1/y.shape[0] * tx.T@e
    return gradient

def compute_gradient_sigmoid(y, tx, w):
    """compute the gradient of loss."""
    grad = tx.T@(sigmoid(tx@w) - y)
    return grad


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    loss = -1
    print("[LSGD] Begin")
    for n_iter in range(max_iters):
        gradient = compute_gradient_MSE(y, tx, w)
        w = w - gamma*gradient
        loss = compute_loss_MSE(y, tx, w)
        if(n_iter % 100 == 0):
            print("[LS-GD] iter {}, loss = {}".format(n_iter, loss))
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    batch_size = 1
    w = initial_w
    loss = -1
    min_loss = 1000
    min_weights = initial_w
    print("[LS-SGD] Begin")
    for n_iter, batches in enumerate(batch_iter(y, tx, batch_size, max_iters)):
        y_i, tx_i = batches
        gradient = compute_gradient_MSE(y_i, tx_i, w)
        w = w - gamma*gradient
        loss = compute_loss_MSE(y, tx, w)
        if(loss < min_loss):
            min_loss = loss
            min_weights = w
        if(n_iter % 100 == 0):
            print("[LS-SGD] iter {}, loss = {}, best_loss = {}".format(n_iter, loss, min_loss))
    return min_weights, min_loss

def sigmoid_GD(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    gradient = compute_gradient_sigmoid(y, tx, w)
    w = w - gamma*gradient
    loss = compute_loss_sigmoid(y, tx, w)
    return w, loss
    
def least_squares(y, tx):
    """Compute the least squares solution."""
    A = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(A, b)
    return w, compute_loss_MSE(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambda_prime = lambda_*2*tx.shape[0]
    A = tx.T@tx + lambda_prime*np.identity(tx.shape[1])
    b = tx.T@y
    w = np.linalg.solve(A, b)
    return w, compute_loss_RMSE(y, tx, w)

def cross_validation(y, x, k_indices, k):
    """return the loss of ridge regression."""
    
    test_x = x[k_indices[k]]
    test_y = y[k_indices[k]]
    train_x = np.delete(x, k_indices[k], 0)
    train_y = np.delete(y, k_indices[k], 0)
    print("{}, {}, {}, {}".format(x.shape, len(k_indices), test_x.shape, train_x.shape))
    
    w_initial = np.zeros((train_x.shape[1], 1))
    w, loss = least_squares_SGD(train_y, train_x, w_initial, 5000, 0.7)
    
    loss_tr = (2*compute_loss_RMSE(train_y, train_x, w))**0.5
    loss_te = (2*compute_loss_RMSE(test_y, test_x, w))**0.5
    return loss_tr, loss_te

def full_cross_validation(y, x):
    seed = 1
    k_fold = 4
    
    k_indices = build_k_indices(y, k_fold, seed)
    
    temp_rmse_tr = 0
    temp_rmse_te = 0
    for i in range(k_fold):
        temp_tr, temp_te = cross_validation(y, x, k_indices, i)
        temp_rmse_tr += temp_tr
        temp_rmse_te += temp_te
    return temp_rmse_tr/k_fold, temp_rmse_te/k_fold

def learning_by_gradient_descent_logistic_regression(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = compute_loss_RMSE(y, tx, w)
    gradient = compute_gradient_sigmoid(y, tx, w)
    w = w - gamma*gradient 
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    losses = []
    w = initial_w
    threshold = 1e-8
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_gradient_descent_logistic_regression(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]

def learning_by_penalized_gradient_descent_logistic_regression(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    correction = 0
    for wi in w:
        correction += wi**2
    correction *= lambda_
    loss = compute_loss_sigmoid(y, tx, w) + correction
    gradient = compute_gradient_sigmoid(y, tx, w) + correction
    w = w - gamma*gradient 
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    losses = []
    w = initial_w
    threshold = 1e-8
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_penalized_gradient_descent_logistic_regression(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]

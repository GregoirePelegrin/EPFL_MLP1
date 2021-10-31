import numpy as np
import matplotlib.pyplot as plt

import datetime
from implementations import *
from proj1_helpers import *

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

# Define the parameters of the algorithm.
batch_size = 1
max_iters = 5000
gamma = 0.7

# Initialization
std_tx = standardize(tX)
std_y = np.asmatrix(y).T
std_tx = np.c_[np.ones((std_tx.shape[0], 1)), std_tx]

def stochastic_gradient_descent(y, tx, batch_size, max_iters, gamma):
    w_initial = np.zeros((tx.shape[1], 1))
    start_time = datetime.datetime.now()
    w, loss = least_squares_SGD(y, tx, w_initial, batch_size, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("Stochastic Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))

    return w, loss

weights, loss = stochastic_gradient_descent(std_y, std_tx, batch_size, max_iters, gamma)

DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = '../data/output.csv'
std_tX_test = standardize(tX_test)
std_tX_test = np.c_[np.ones((std_tX_test.shape[0], 1)), std_tX_test]
y_pred = predict_labels(weights, std_tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
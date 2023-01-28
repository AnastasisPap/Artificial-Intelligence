from random import sample
from time import sleep
from threading import Thread
from tqdm import tqdm
from utils import *
from MLP import rnn
import log_reg
import numpy as np
from sklearn.model_selection import train_test_split 
from naive_bayes import *
import itertools

# Hyperparameters:
# n, m, k
# Logistic regression: learning_rate, lambda
n = k = 15000
def generate_random_numbers(n, low, upper, step):
    nums = [round(val, 4) for val in list(np.arange(low, upper + step, step))]
    return sorted(sample(nums, n))

def get_combinations(n_m, n_learning_rate, n_lambda):
    random_m = generate_random_numbers(n_m, 1000, 20000, 1000)

    random_learning_rates = generate_random_numbers(n_learning_rate, 0.005, 0.05, 0.001)
    random_lambda = generate_random_numbers(n_lambda, 0.001, 0.01, 0.001)
    combinations = list(itertools.product(random_m, random_learning_rates, random_lambda))

    return combinations

def thread_search(combinations):
    print('Starting')
    min_mean_error = 1
    best_combination = None
    val_size = 0.2

    for combination in combinations:
        m, lr, l = combination
        train, _ = preprocess_data(n, m, k)

        x_train, x_test , y_train, y_test = train_test_split(train[1], train[2], test_size=val_size)
        bayes_error = evaluate_bayes((x_train, y_train), (x_test, y_test), 100, True)
        log_reg_error = log_reg.evaluate_logistic_regression((x_train, y_train), (x_test, y_test), 100, 100, lr, l, True)
        #x_train, x_test, y_train, y_test = train_test_split(train[0], train[2], test_size=val_size)
        #rnn_error = rnn((x_train, y_train), (x_test, y_test), m, 100, 10, True)

        curr_min_error = (bayes_error + log_reg_error) / 2
        if curr_min_error < min_mean_error:
            min_mean_error = curr_min_error
            best_combination = (n, k, m, lr, l)
    
    return 1 - min_mean_error, best_combination
    

class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    
    def join(self):
        Thread.join(self)
        return self._return
    
def test(combinations):
    return combinations

def search(num_of_threads):
    combinations = get_combinations(1, 2, 2)

    threads = []
    step = len(combinations) // num_of_threads
    for i in range(num_of_threads):
        if i != num_of_threads - 1: curr_combinations = combinations[i*step : (i+1)*step]
        else: curr_combinations = combinations[i*step:]
        threads.append(CustomThread(target=thread_search, args=[curr_combinations]))

    for thread in threads:
        thread.start()
    
    results = [thread.join(timeout=3600) for thread in threads]
    min_idx = results.index(max(results, key=lambda x: x[0]))
    print(results[min_idx])

search(4)
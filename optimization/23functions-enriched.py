import numpy as np

# Define the Shekel function for reuse
def shekel(x, m):
    d = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    a = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                  [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                  [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    return -np.sum([1 / (np.dot(x - a[i], x - a[i]) + d[i]) for i in range(m)])

# Comprehensive dictionary of benchmark functions
benchmark_functions = {
    "Sphere": {
        "function": lambda x: np.sum(x**2),
        "bounds": [[-100, 100]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Schwefel_P2": {
        "function": lambda x: np.sum(np.abs(x)) + np.prod(np.abs(x)),
        "bounds": [[-10, 10]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Schwefel_P1_2": {
        "function": lambda x: np.sum([np.sum(x[:i+1])**2 for i in range(len(x))]),
        "bounds": [[-100, 100]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Step": {
        "function": lambda x: np.sum(np.floor(x + 0.5)**2),
        "bounds": [[-100, 100]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Rosenbrock": {
        "function": lambda x: np.sum([100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1)]),
        "bounds": [[-30, 30]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Quartic_with_noise": {
        "function": lambda x: np.sum([(i + 1) * x[i]**4 for i in range(len(x))]) + np.random.uniform(0, 1),
        "bounds": [[-1.28, 1.28]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Schwefel_P2_21": {
        "function": lambda x: np.max(np.abs(x)),
        "bounds": [[-100, 100]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Rastrigin": {
        "function": lambda x: np.sum([x_i**2 - 10 * np.cos(2 * np.pi * x_i) + 10 for x_i in x]),
        "bounds": [[-5.12, 5.12]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Ackley": {
        "function": lambda x: -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e,
        "bounds": [[-32, 32]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Griewank": {
        "function": lambda x: 1 + np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))),
        "bounds": [[-600, 600]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Penalized_1": {
        "function": test_func_12,
        "bounds": [[-50, 50]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Penalized_2": {
        "function": test_func_13,
        "bounds": [[-50, 50]] * 30,
        "dimensionality": 30,
        "global_min": 0
    },
    "Shekel_Foxholes": {
        "function": test_func_14,
        "bounds": [[-65.536, 65.536]] * 2,
        "dimensionality": 2,
        "global_min": 1
    },
    "Kowalik": {
        "function": test_func_15,
        "bounds": [[-5, 5]] * 4,
        "dimensionality": 4,
        "global_min": 0.0003075
    },
    "Six_Hump_Camel": {
        "function": test_func_16,
        "bounds": [[-5, 5]] * 2,
        "dimensionality": 2,
        "global_min": -1.0316285
    },
    "Branin": {
        "function": test_func_17,
        "bounds": [[-5, 10], [0, 15]],
        "dimensionality": 2,
        "global_min": 0.398
    },
    "Goldstein_Price": {
        "function": test_func_18,
        "bounds": [[-2, 2]] * 2,
        "dimensionality": 2,
        "global_min": 3
    },
    "Hartman_3": {
        "function": test_func_19,
        "bounds": [[0, 1]] * 3,
        "dimensionality": 3,
        "global_min": -3.86
    },
    "Hartman_6": {
        "function": test_func_20,
        "bounds": [[0, 1]] * 6,
        "dimensionality": 6,
        "global_min": -3.32
    },
    "Shekel_5": {
        "function": test_func_21,
        "bounds": [[0, 10]] * 4,
        "dimensionality": 4,
        "global_min": -10
    },
    "Shekel_7": {
        "function": test_func_22,
        "bounds": [[0, 10]] * 4,
        "dimensionality": 4,
        "global_min": -10
    },
    "Shekel_10": {
        "function": test_func_23,
        "bounds": [[0, 10]] * 4,
        "dimensionality": 4,
        "global_min": -10
    }
}

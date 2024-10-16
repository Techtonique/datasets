import numpy as np

def test_func_1(x):
    return np.sum(x**2)

def test_func_2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def test_func_3(x):
    return np.sum([np.sum(x[:i+1])**2 for i in range(len(x))])

def test_func_4(x):
    return np.max(np.abs(x))

def test_func_5(x):
    return np.sum([100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(len(x) - 1)])

def test_func_6(x):
    return np.sum(np.floor(x + 0.5)**2)

def test_func_7(x):
    return np.sum([(i + 1) * x[i]**4 for i in range(len(x) - 1)]) + np.random.uniform(0, 1)

def test_func_8(x):
    return -np.sum([x_i * np.sin(np.sqrt(np.abs(x_i))) for x_i in x])

def test_func_9(x):
    return np.sum([x_i**2 - 10 * np.cos(2 * np.pi * x_i) + 10 for x_i in x])

def test_func_10(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e

def test_func_11(x):
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def test_func_12(x):
    y = 1 + (x + 1) / 4
    u = lambda z, a, k, m: k * np.maximum(0, np.abs(z) - a)**m
    return (np.pi / len(x)) * (10 * np.sin(np.pi * y[0])**2 + np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2)) + (y[-1] - 1)**2) + np.sum(u(x, 10, 100, 4))

def test_func_13(x):
    u = lambda z, a, k, m: k * np.maximum(0, np.abs(z) - a)**m
    return 0.1 * (np.sin(3 * np.pi * x[0])**2 + np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2)) + (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)) + np.sum(u(x, 5, 100, 4))

def test_func_14(x):
    a = np.array([
        [-32, -16, 0, 16, 32] * 5,
        [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5
    ])
    return 1 / (1 / 500 + np.sum(1 / (np.arange(1, 26) + np.sum((x - a)**6, axis=0))))

def test_func_15(x):
    a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    b = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    return np.sum([(a[i] - x[0] * (b[i]**2 + b[i] * x[1]) / (b[i]**2 + b[i] * x[2] + x[3]))**2 for i in range(11)])

def test_func_16(x):
    return 4 * (x[0]**2 - x[1]**2 + x[1]**4) - 2.1 * x[0]**4 + x[0]**6 / 3 + x[0] * x[1]

def test_func_17(x):
    return (x[1] - (5.1 / (4 * np.pi**2)) * x[0]**2 + 5 * x[0] / np.pi - 6)**2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

def test_func_18(x):
    return (1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * x[0]**2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)) * (30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * x[0]**2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1]**2))

def test_func_19(x):
    d = np.array([1, 1.2, 3, 3.2])
    a = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    p = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    return -np.sum([d[i] * np.exp(-np.sum(a[i] * (x - p[i])**2)) for i in range(4)])

def test_func_20(x):
    d = np.array([1, 1.2, 3, 3.2])
    a = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    p = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991], [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.665], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    return -np.sum([d[i] * np.exp(-np.sum(a[i] * (x - p[i])**2)) for i in range(4)])

def shekel(x, m):
    d = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    a = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    return -np.sum([1 / (np.dot(x - a[i], x - a[i]) + d[i]) for i in range(m)])

def test_func_21(x):
    return shekel(x, 5)

def test_func_22(x):
    return shekel(x, 7)

def test_func_23(x):
    return shekel(x, 10)

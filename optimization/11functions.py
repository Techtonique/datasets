import numpy as np

# [−5,10], min = 0 
def rosenbrock(x):
    return sum(100.0 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

# [−32.768,32.768], min = 0
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = -a * np.exp(-b * np.sqrt(sum(xi**2 for xi in x) / d))
    sum2 = -np.exp(sum(np.cos(c * xi) for xi in x) / d)
    return a + np.exp(1) + sum1 + sum2

# [−512, 512], min = -959.6407
def eggholder(x):
    x1 = x[0]
    x2 = x[1]
    term1 = -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1/2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))
    return term1 + term2


# [−5.12, 5.12], min = 0
def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + sum((xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x)

# [−10, 10], min = 0
def levy(x):
    w = [(xi - 1) / 4 + 1 for xi in x]
    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    terms = sum((wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2) for wi in w[:-1])
    return term1 + terms + term3

# [−600, 600], min = 0
def griewank(x):
    sum_term = sum((xi**2) / 4000 for xi in x)
    prod_term = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
    return sum_term - prod_term + 1


# Himmelblau's Function
def himmelblau(x):
    """
    Himmelblau's Function:
    - Global minima located at:
      (3.0, 2.0),
      (-2.805118, 3.131312),
      (-3.779310, -3.283186),
      (3.584428, -1.848126)
    - Function value at all minima: f(x) = 0
    """
    x1 = x[0]
    x2 = x[1]
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

# Six-Hump Camel Function
def six_hump_camel(x):
    """
    Six-Hump Camel Function:
    - Global minima located at:
      (0.0898, -0.7126),
      (-0.0898, 0.7126)
    - Function value at the minima: f(x) = -1.0316
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2
    return term1 + term2 + term3

# Michalewicz Function
def michalewicz(x, m=10):
    """
    Michalewicz Function (for n=2 dimensions):
    - Global minimum located at approximately: (2.20, 1.57)
    - Function value at the minimum: f(x) ≈ -1.8013
    """
    return -sum(np.sin(xi) * (np.sin((i + 1) * xi**2 / np.pi))**(2 * m) for i, xi in enumerate(x))

# Goldstein-Price Function
def goldstein_price(x):
    """
    Goldstein-Price Function:
    - Global minimum located at: (0, -1)
    - Function value at the minimum: f(x) = 3
    """
    x1 = x[0]
    x2 = x[1]
    term1 = (1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2))
    term2 = (30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2))
    return term1 * term2

# Booth Function
def booth(x):
    """
    Booth Function:
    - Global minimum located at: (1, 3)
    - Function value at the minimum: f(x) = 0
    """
    x1 = x[0]
    x2 = x[1]
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2
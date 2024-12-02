import numpy as np
from numpy.random import default_rng

def frank_wolfe_tufw_optimized(
        loss_fn, grad_fn, hessian_fn, init_x, lmo_fn, dataset, step_size_fn, num_iterations
):
    """
    Taylor-Updated Frank-Wolfe (TUFW) Optimization Algorithm.

    Parameters:
        loss_fn: Callable
            Function to compute the loss, accepts features, labels, and solution vector.
        grad_fn: Callable
            Function to compute the gradient for a single data point.
        hessian_fn: Callable
            Function to compute the Hessian for a single data point.
        init_x: np.ndarray
            Initial solution vector.
        lmo_fn: Callable
            Linear Minimization Oracle (LMO) to solve sub-problems.
        dataset: Tuple (features, labels)
            Features and labels for the dataset.
        step_size_fn: Callable
            Function to compute step size for each iteration.
        num_iterations: int
            Number of iterations to run the algorithm.

    Returns:
        x: np.ndarray
            Final solution vector.
        loss_history: list
            History of loss values over iterations.
    """
    # Initialize variables
    features, labels = dataset
    n, p = features.shape
    x = init_x.copy()  # Current solution vector
    bi = np.tile(init_x, (n, 1))  # Taylor points initialized to x0
    qk = np.zeros(p)  # Precomputed gradient component (Proposition 2.1)
    Hk = np.zeros((p, p))  # Precomputed Hessian component (Proposition 2.1)
    loss_history = []

    rng = default_rng()  # Random generator for stochastic batch selection

    # Precompute initial qk and Hk
    for i in range(n):
        grad_i = grad_fn(features[i], labels[i], bi[i])
        hess_i = hessian_fn(features[i], labels[i], bi[i])
        qk += grad_i - hess_i @ bi[i]
        Hk += hess_i
    qk /= n
    Hk /= n

    # Start Frank-Wolfe iterations
    for k in range(num_iterations):
        # Step 1: Stochastic Taylor-point update (Rule - SBD)
        beta_k = int(n / np.sqrt(k + 1))  # |Bk| decreases as O(1/sqrt(k))
        Bk = rng.choice(n, size=beta_k, replace=False)  # Random subset of indices

        for i in Bk:
            grad_i = grad_fn(features[i], labels[i], x)
            hess_i = hessian_fn(features[i], labels[i], x)

            # Incrementally update qk and Hk
            prev_grad = grad_fn(features[i], labels[i], bi[i])
            prev_hess = hessian_fn(features[i], labels[i], bi[i])
            qk += (grad_i - hess_i @ x) - (prev_grad - prev_hess @ bi[i])
            Hk += (hess_i - prev_hess)

            # Update Taylor points
            bi[i] = x

        # Step 2: Compute gradient efficiently using Proposition 2.1
        gk = qk + Hk @ x

        # Step 3: Solve LMO (Linear Minimization Oracle)
        sk = lmo_fn(gk)

        # Step 4: Update feasible solution
        step_size = step_size_fn(k)
        x = x + step_size * (sk - x)

        # Record loss for analysis
        loss_history.append(loss_fn(features, labels, x))

    return x, loss_history

# Taylor-Updated Frank-Wolfe (TUFW) Optimization Algorithm

The **Taylor-Updated Frank-Wolfe (TUFW)** algorithm is an advanced optimization method tailored for large-scale empirical risk minimization (ERM) problems. This implementation includes stochastic updates for Taylor points and efficient gradient computation based on second-order approximations.

---

## Features

- **Gradient Approximation**: Reduces computation by leveraging Taylor-expanded gradients.
- **Dynamic Batch Updates**: Implements Rule-SBD for batch-size selection.
- **Scalability**: Handles large datasets efficiently.
- **Versatility**: Designed for both convex and nonconvex optimization problems.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/<your_username>/frank-wolfe-tufw.git
    cd frank-wolfe-tufw
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---


---

## How to Run

1. Prepare your environment:
    - Ensure Python 3.7+ is installed.
    - Install dependencies as shown in the **Installation** section.

2. Navigate to the `src/` directory and run the TUFW example:
    ```bash
    python frank_wolfe_tufw.py
    ```

3. Example output:
    ```
    Final solution: [ 0.213 -0.874  0.345 ... ]
    Loss history: [0.693, 0.578, 0.481, ...]
    ```

---

## Example Usage

Here’s an example to run the TUFW algorithm on a synthetic dataset:

```python
import numpy as np
from src.frank_wolfe_tufw import frank_wolfe_tufw_optimized
from src.utils import logistic_loss, logistic_grad, logistic_hessian, lmo_l1_ball

# Generate synthetic dataset
np.random.seed(0)
n, p = 100, 10
features = np.random.randn(n, p)
labels = np.random.choice([-1, 1], size=n)
init_x = np.zeros(p)

# Run TUFW
final_x, history = frank_wolfe_tufw_optimized(
    loss_fn=logistic_loss,
    grad_fn=logistic_grad,
    hessian_fn=logistic_hessian,
    init_x=init_x,
    lmo_fn=lmo_l1_ball,
    dataset=(features, labels),
    step_size_fn=lambda k: 2 / (k + 2),
    num_iterations=50,
)

print("Final solution:", final_x)
print("Loss history:", history)
```

## Analyzing the Output

- **Final Solution**: The final solution vector, `final_x`, represents the optimized parameters for the loss function.
- **Loss History**: The `history` array tracks the value of the loss function at each iteration, which should ideally decrease over time.

---

## Algorithm Details

The Taylor-Updated Frank-Wolfe (TUFW) algorithm is based on:
1. **Taylor-Approximated Gradients**: Using Taylor expansion to approximate gradients and reduce computational overhead.
2. **Dynamic Batch Updates**: Stochastic updates of Taylor points, reducing the number of gradient and Hessian computations.
3. **Linear Minimization Oracle (LMO)**: Efficient subproblem solving for constraints such as the L1 ball.

### Key Steps
1. **Initialize**:
    - Start with an initial solution vector \(x_0\) and Taylor points initialized to the same value.
2. **Stochastic Updates**:
    - Select a subset of data points for Taylor point updates, reducing computations.
3. **Gradient Computation**:
    - Efficiently compute the gradient using precomputed terms (Proposition 2.1 from the referenced paper).
4. **Linear Minimization**:
    - Solve the subproblem to find the descent direction.
5. **Update Solution**:
    - Use a step-size schedule to update the solution vector.
6. **Iterate**:
    - Repeat the process for a predefined number of iterations.

---

## Contribution and Development

This repository is an implementation of the TUFW algorithm inspired by the paper:  
[Using Taylor-Approximated Gradients to Improve the Frank-Wolfe Method](https://epubs.siam.org/doi/10.1137/22M1519286).
****
Developed by **Darshil Patel**.  
Feel free to contribute to this repository by submitting issues or pull requests on GitHub.

## Contact

For questions, suggestions, or collaboration:
- Email: darshilamit@gmail.com
- GitHub: [LED-0102](https://github.com/LED-0102)



import numpy as np
from src.frank_wolfe_tufw import frank_wolfe_tufw_optimized
from src.utils import logistic_loss, logistic_grad, logistic_hessian, lmo_l1_ball

def test_logistic_loss():
    """Test the logistic loss function."""
    features = np.array([[1, 2], [3, 4]])
    labels = np.array([1, -1])
    x = np.array([0.5, -0.5])
    loss = logistic_loss(features, labels, x)
    assert loss > 0, "Loss should be positive"

def test_linear_minimization_oracle():
    """Test the LMO for L1 ball."""
    gradient = np.array([3, -1, 0.5])
    s = lmo_l1_ball(gradient, lambda_val=1)
    assert np.sum(np.abs(s)) <= 1, "Solution should be within the L1 ball"
    assert np.count_nonzero(s) == 1, "Solution should be sparse"

def test_frank_wolfe():
    """Test the main TUFW algorithm on a small dataset."""
    # Small synthetic dataset
    np.random.seed(0)
    features = np.random.randn(10, 5)
    labels = np.random.choice([-1, 1], size=10)
    init_x = np.zeros(5)

    # Run TUFW
    final_x, history = frank_wolfe_tufw_optimized(
        loss_fn=logistic_loss,
        grad_fn=logistic_grad,
        hessian_fn=logistic_hessian,
        init_x=init_x,
        lmo_fn=lmo_l1_ball,
        dataset=(features, labels),
        step_size_fn=lambda k: 2 / (k + 2),
        num_iterations=10,
    )

    # Validate results
    assert len(history) == 10, "Loss history should have one entry per iteration"
    assert np.linalg.norm(final_x) > 0, "Final solution should not be zero"

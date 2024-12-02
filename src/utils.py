import numpy as np

def logistic_loss(features, labels, x):
    """Logistic regression loss function."""
    logits = features @ x
    return np.mean(np.log(1 + np.exp(-labels * logits)))

def logistic_grad(features_i, labels_i, x):
    """Gradient of logistic regression loss for a single data point."""
    logits = np.dot(features_i, x)
    grad = -labels_i * features_i / (1 + np.exp(labels_i * logits))
    return grad

def logistic_hessian(features_i, labels_i, x):
    """Hessian of logistic regression loss for a single data point."""
    logits = np.dot(features_i, x)
    sigmoid = 1 / (1 + np.exp(-logits))
    hessian = np.outer(features_i, features_i) * sigmoid * (1 - sigmoid)
    return hessian

def lmo_l1_ball(gradient, lambda_val=1):
    """Linear Minimization Oracle for an L1 ball."""
    s = np.zeros_like(gradient)
    idx = np.argmax(np.abs(gradient))
    s[idx] = -lambda_val * np.sign(gradient[idx])
    return s

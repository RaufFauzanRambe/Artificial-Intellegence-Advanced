"""
Optimization Module - Gradient-Based Optimization Algorithms from Scratch.

Implements various gradient descent optimizers for minimizing scalar functions:
  - Vanilla Gradient Descent
  - Gradient Descent with Momentum
  - Adam (Adaptive Moment Estimation)
  - RMSprop

Includes:
  - Test functions (Rosenbrock, Rastrigin, Sphere, Beale)
  - Learning rate schedulers (step decay, exponential decay, cosine annealing)
  - Numerical gradient computation for functions without analytic gradients

Example:
    >>> from optimization import AdamOptimizer, rosenbrock
    >>> optimizer = AdamOptimizer(learning_rate=0.01)
    >>> result = optimizer.optimize(rosenbrock, np.array([-1.0, 1.0]))
    >>> print(f"Minimum at: {result['x']}")
"""

import numpy as np
from typing import Callable, Optional


# =============================================================================
# Test Functions
# =============================================================================

def sphere(x: np.ndarray) -> float:
    """Sphere function: f(x) = sum(x_i^2).

    Global minimum at x = [0, 0, ...] with f(x) = 0.

    Args:
        x: Input vector.

    Returns:
        Function value.
    """
    return np.sum(x ** 2)


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function (banana function).

    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2)

    Global minimum at x = [1, 1, ...] with f(x) = 0.
    Known to be challenging for gradient-based optimization due to a
    narrow curved valley.

    Args:
        x: Input vector (typically 2-D).

    Returns:
        Function value.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Analytical gradient of the Rosenbrock function.

    Args:
        x: Input vector.

    Returns:
        Gradient vector.
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for i in range(len(x) - 1):
        grad[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
        grad[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    return grad


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function.

    f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))

    Global minimum at x = [0, 0, ...] with f(x) = 0.
    Highly multimodal with many local minima, making it a challenging
    test for optimization algorithms.

    Args:
        x: Input vector.

    Returns:
        Function value.
    """
    A = 10.0
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2.0 * np.pi * x))


def rastrigin_grad(x: np.ndarray) -> np.ndarray:
    """Analytical gradient of the Rastrigin function.

    Args:
        x: Input vector.

    Returns:
        Gradient vector.
    """
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


def beale(x: np.ndarray) -> float:
    """Beale function (2-D only).

    f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2

    Global minimum at x = [3, 0.5] with f(x) = 0.

    Args:
        x: 2-D input vector [x, y].

    Returns:
        Function value.
    """
    x_val, y_val = x[0], x[1]
    term1 = (1.5 - x_val + x_val * y_val) ** 2
    term2 = (2.25 - x_val + x_val * y_val ** 2) ** 2
    term3 = (2.625 - x_val + x_val * y_val ** 3) ** 2
    return term1 + term2 + term3


def numerical_gradient(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Compute the gradient of a function numerically using central differences.

    Useful for functions that don't have an analytical gradient.

    Args:
        f: Scalar function to differentiate.
        x: Point at which to evaluate the gradient.
        h: Finite difference step size.

    Returns:
        Numerical gradient vector of the same shape as x.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * h)
    return grad


# =============================================================================
# Learning Rate Schedulers
# =============================================================================

class StepDecay:
    """Step decay learning rate scheduler.

    Multiplies the learning rate by `drop_factor` every `step_size` epochs.
    LR(t) = initial_lr * drop_factor^(floor(t / step_size))

    Attributes:
        initial_lr: Starting learning rate.
        drop_factor: Multiplicative factor applied at each step.
        step_size: Number of iterations between each decay.
    """

    def __init__(self, initial_lr: float, drop_factor: float = 0.5, step_size: int = 100):
        """Initialize the step decay scheduler.

        Args:
            initial_lr: Starting learning rate.
            drop_factor: Factor to multiply LR by at each step.
            step_size: Iterations between decay steps.
        """
        self.initial_lr = initial_lr
        self.drop_factor = drop_factor
        self.step_size = step_size

    def get_lr(self, iteration: int) -> float:
        """Get the learning rate for a given iteration.

        Args:
            iteration: Current iteration number.

        Returns:
            Scheduled learning rate.
        """
        return self.initial_lr * (self.drop_factor ** np.floor(iteration / self.step_size))


class ExponentialDecay:
    """Exponential decay learning rate scheduler.

    LR(t) = initial_lr * exp(-decay_rate * t)

    Attributes:
        initial_lr: Starting learning rate.
        decay_rate: Exponential decay constant.
    """

    def __init__(self, initial_lr: float, decay_rate: float = 0.01):
        """Initialize the exponential decay scheduler.

        Args:
            initial_lr: Starting learning rate.
            decay_rate: Rate of exponential decay.
        """
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate

    def get_lr(self, iteration: int) -> float:
        """Get the learning rate for a given iteration.

        Args:
            iteration: Current iteration number.

        Returns:
            Scheduled learning rate.
        """
        return self.initial_lr * np.exp(-self.decay_rate * iteration)


class CosineAnnealing:
    """Cosine annealing learning rate scheduler.

    LR(t) = eta_min + 0.5 * (initial_lr - eta_min) * (1 + cos(pi * t / T_max))

    Attributes:
        initial_lr: Starting learning rate.
        eta_min: Minimum learning rate.
        T_max: Maximum number of iterations.
    """

    def __init__(self, initial_lr: float, eta_min: float = 0.0, T_max: int = 1000):
        """Initialize the cosine annealing scheduler.

        Args:
            initial_lr: Starting learning rate.
            eta_min: Minimum learning rate.
            T_max: Maximum number of iterations for one cycle.
        """
        self.initial_lr = initial_lr
        self.eta_min = eta_min
        self.T_max = T_max

    def get_lr(self, iteration: int) -> float:
        """Get the learning rate for a given iteration.

        Args:
            iteration: Current iteration number.

        Returns:
            Scheduled learning rate.
        """
        return self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (
            1.0 + np.cos(np.pi * iteration / self.T_max)
        )


# =============================================================================
# Optimizers
# =============================================================================

class GradientDescent:
    """Vanilla Gradient Descent optimizer.

    Update rule:
        x_{t+1} = x_t - lr * grad(f(x_t))

    Attributes:
        learning_rate: Step size for parameter updates.
        history: List of (x, f(x)) tuples at each iteration.
    """

    def __init__(self, learning_rate: float = 0.01):
        """Initialize Gradient Descent.

        Args:
            learning_rate: Learning rate (step size).
        """
        self.learning_rate = learning_rate
        self.history = []

    def optimize(
        self,
        f: Callable,
        x0: np.ndarray,
        grad: Optional[Callable] = None,
        n_iterations: int = 1000,
        tolerance: float = 1e-8,
        verbose: bool = False,
    ) -> dict:
        """Minimize a function using gradient descent.

        Args:
            f: Objective function to minimize.
            x0: Initial point.
            grad: Gradient function. If None, uses numerical gradients.
            n_iterations: Maximum number of iterations.
            tolerance: Stop if gradient norm falls below this.
            verbose: If True, print progress every 100 iterations.

        Returns:
            Dictionary with keys: 'x' (optimal point), 'f' (optimal value),
            'history' (list of function values), 'converged' (bool),
            'iterations' (number of iterations run).
        """
        x = np.array(x0, dtype=np.float64)
        self.history = []

        if grad is None:
            grad_func = lambda xi: numerical_gradient(f, xi)
        else:
            grad_func = grad

        for i in range(n_iterations):
            f_val = f(x)
            self.history.append(f_val)

            g = grad_func(x)
            grad_norm = np.linalg.norm(g)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Iter {i + 1:4d} | f(x) = {f_val:.6e} | "
                      f"||grad|| = {grad_norm:.6e} | x = {x}")

            # Check convergence
            if grad_norm < tolerance:
                if verbose:
                    print(f"  Converged at iteration {i + 1} (gradient norm < {tolerance})")
                return {
                    "x": x,
                    "f": f(x),
                    "history": self.history,
                    "converged": True,
                    "iterations": i + 1,
                }

            x = x - self.learning_rate * g

        return {
            "x": x,
            "f": f(x),
            "history": self.history,
            "converged": False,
            "iterations": n_iterations,
        }


class GradientDescentWithMomentum:
    """Gradient Descent with Momentum.

    Update rules:
        v_{t+1} = momentum * v_t - lr * grad(f(x_t))
        x_{t+1} = x_t + v_{t+1}

    Momentum helps accelerate gradient descent in the relevant direction
    and dampens oscillations.

    Attributes:
        learning_rate: Step size for parameter updates.
        momentum: Momentum coefficient (typically 0.9).
        velocity: Accumulated velocity vector.
        history: List of function values at each iteration.
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """Initialize Gradient Descent with Momentum.

        Args:
            learning_rate: Learning rate.
            momentum: Momentum coefficient in [0, 1).
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.history = []

    def optimize(
        self,
        f: Callable,
        x0: np.ndarray,
        grad: Optional[Callable] = None,
        n_iterations: int = 1000,
        tolerance: float = 1e-8,
        verbose: bool = False,
    ) -> dict:
        """Minimize a function using gradient descent with momentum.

        Args:
            f: Objective function to minimize.
            x0: Initial point.
            grad: Gradient function. If None, uses numerical gradients.
            n_iterations: Maximum number of iterations.
            tolerance: Stop if gradient norm falls below this.
            verbose: If True, print progress.

        Returns:
            Dictionary with optimization results.
        """
        x = np.array(x0, dtype=np.float64)
        self.velocity = np.zeros_like(x)
        self.history = []

        if grad is None:
            grad_func = lambda xi: numerical_gradient(f, xi)
        else:
            grad_func = grad

        for i in range(n_iterations):
            f_val = f(x)
            self.history.append(f_val)

            g = grad_func(x)
            grad_norm = np.linalg.norm(g)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Iter {i + 1:4d} | f(x) = {f_val:.6e} | "
                      f"||grad|| = {grad_norm:.6e} | x = {x}")

            if grad_norm < tolerance:
                if verbose:
                    print(f"  Converged at iteration {i + 1}")
                return {
                    "x": x, "f": f(x), "history": self.history,
                    "converged": True, "iterations": i + 1,
                }

            self.velocity = self.momentum * self.velocity - self.learning_rate * g
            x = x + self.velocity

        return {
            "x": x, "f": f(x), "history": self.history,
            "converged": False, "iterations": n_iterations,
        }


class AdamOptimizer:
    """Adam (Adaptive Moment Estimation) optimizer.

    Combines momentum (first moment) and adaptive learning rates (second moment).

    Update rules:
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m_hat = m_t / (1 - beta1^t)     (bias correction)
        v_hat = v_t / (1 - beta2^t)     (bias correction)
        x_{t+1} = x_t - lr * m_hat / (sqrt(v_hat) + eps)

    Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2015).

    Attributes:
        learning_rate: Step size.
        beta1: Exponential decay rate for first moment estimates.
        beta2: Exponential decay rate for second moment estimates.
        epsilon: Small constant for numerical stability.
        m: First moment (momentum) vector.
        v: Second moment (adaptive LR) vector.
        t: Time step counter.
        history: List of function values.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """Initialize the Adam optimizer.

        Args:
            learning_rate: Learning rate (alpha in the paper).
            beta1: Decay rate for first moment (default 0.9).
            beta2: Decay rate for second moment (default 0.999).
            epsilon: Small constant for numerical stability (default 1e-8).
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        self.history = []

    def optimize(
        self,
        f: Callable,
        x0: np.ndarray,
        grad: Optional[Callable] = None,
        n_iterations: int = 1000,
        tolerance: float = 1e-8,
        verbose: bool = False,
    ) -> dict:
        """Minimize a function using Adam.

        Args:
            f: Objective function to minimize.
            x0: Initial point.
            grad: Gradient function. If None, uses numerical gradients.
            n_iterations: Maximum number of iterations.
            tolerance: Stop if gradient norm falls below this.
            verbose: If True, print progress.

        Returns:
            Dictionary with optimization results.
        """
        x = np.array(x0, dtype=np.float64)
        self.m = np.zeros_like(x)
        self.v = np.zeros_like(x)
        self.t = 0
        self.history = []

        if grad is None:
            grad_func = lambda xi: numerical_gradient(f, xi)
        else:
            grad_func = grad

        for i in range(n_iterations):
            self.t += 1
            f_val = f(x)
            self.history.append(f_val)

            g = grad_func(x)
            grad_norm = np.linalg.norm(g)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Iter {i + 1:4d} | f(x) = {f_val:.6e} | "
                      f"||grad|| = {grad_norm:.6e} | x = {x.round(6)}")

            if grad_norm < tolerance:
                if verbose:
                    print(f"  Converged at iteration {i + 1}")
                return {
                    "x": x, "f": f(x), "history": self.history,
                    "converged": True, "iterations": i + 1,
                }

            # Update biased first and second moment estimates
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * g
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * g ** 2

            # Bias-corrected estimates
            m_hat = self.m / (1.0 - self.beta1 ** self.t)
            v_hat = self.v / (1.0 - self.beta2 ** self.t)

            # Parameter update
            x = x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return {
            "x": x, "f": f(x), "history": self.history,
            "converged": False, "iterations": n_iterations,
        }


class RMSpropOptimizer:
    """RMSprop optimizer.

    Maintains a moving average of squared gradients and divides the learning
    rate by the root of this average.

    Update rules:
        cache_t = decay * cache_{t-1} + (1 - decay) * grad^2
        x_{t+1} = x_t - lr * grad / (sqrt(cache_t) + eps)

    Reference: "Generating Sequences With Recurrent Neural Networks" (Sutskever, 2013).

    Attributes:
        learning_rate: Step size.
        decay: Decay rate for the moving average.
        epsilon: Small constant for numerical stability.
        cache: Moving average of squared gradients.
        history: List of function values.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0.9,
        epsilon: float = 1e-8,
    ):
        """Initialize the RMSprop optimizer.

        Args:
            learning_rate: Learning rate.
            decay: Decay rate for the squared gradient moving average.
            epsilon: Small constant for numerical stability.
        """
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.cache = None
        self.history = []

    def optimize(
        self,
        f: Callable,
        x0: np.ndarray,
        grad: Optional[Callable] = None,
        n_iterations: int = 1000,
        tolerance: float = 1e-8,
        verbose: bool = False,
    ) -> dict:
        """Minimize a function using RMSprop.

        Args:
            f: Objective function to minimize.
            x0: Initial point.
            grad: Gradient function. If None, uses numerical gradients.
            n_iterations: Maximum number of iterations.
            tolerance: Stop if gradient norm falls below this.
            verbose: If True, print progress.

        Returns:
            Dictionary with optimization results.
        """
        x = np.array(x0, dtype=np.float64)
        self.cache = np.zeros_like(x)
        self.history = []

        if grad is None:
            grad_func = lambda xi: numerical_gradient(f, xi)
        else:
            grad_func = grad

        for i in range(n_iterations):
            f_val = f(x)
            self.history.append(f_val)

            g = grad_func(x)
            grad_norm = np.linalg.norm(g)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Iter {i + 1:4d} | f(x) = {f_val:.6e} | "
                      f"||grad|| = {grad_norm:.6e} | x = {x.round(6)}")

            if grad_norm < tolerance:
                if verbose:
                    print(f"  Converged at iteration {i + 1}")
                return {
                    "x": x, "f": f(x), "history": self.history,
                    "converged": True, "iterations": i + 1,
                }

            self.cache = self.decay * self.cache + (1.0 - self.decay) * g ** 2
            x = x - self.learning_rate * g / (np.sqrt(self.cache) + self.epsilon)

        return {
            "x": x, "f": f(x), "history": self.history,
            "converged": False, "iterations": n_iterations,
        }


# =============================================================================
# Utility Functions
# =============================================================================

def compare_optimizers(
    f: Callable,
    grad: Callable,
    x0: np.ndarray,
    n_iterations: int = 500,
    true_minimum: float = 0.0,
    name: str = "Function",
) -> None:
    """Compare multiple optimizers on the same function.

    Args:
        f: Objective function.
        grad: Analytical gradient function.
        x0: Starting point.
        n_iterations: Number of iterations for each optimizer.
        true_minimum: Known minimum value (for reference).
        name: Display name of the function.
    """
    print(f"\n  Function: {name}")
    print(f"  Starting point: {x0}")
    print(f"  Known minimum:  {true_minimum}")
    print(f"  Iterations:     {n_iterations}")
    print()

    optimizers = [
        ("Vanilla GD (lr=0.001)", GradientDescent(learning_rate=0.001)),
        ("Momentum (lr=0.001, m=0.9)", GradientDescentWithMomentum(learning_rate=0.001, momentum=0.9)),
        ("Adam (lr=0.1)", AdamOptimizer(learning_rate=0.1)),
        ("RMSprop (lr=0.01)", RMSpropOptimizer(learning_rate=0.01)),
    ]

    results = []
    for opt_name, optimizer in optimizers:
        result = optimizer.optimize(f, x0, grad=grad, n_iterations=n_iterations)
        results.append((opt_name, result))
        print(f"  {opt_name}:")
        print(f"    Final x = {result['x'].round(6)}")
        print(f"    Final f(x) = {result['f']:.8e}")
        print(f"    Converged: {result['converged']}")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("Optimization Module - Algorithm Demos")
    print("=" * 60)

    # --- 1. Rosenbrock Function ---
    print("\n1. Rosenbrock Function Optimization")
    print("-" * 60)
    compare_optimizers(
        f=rosenbrock,
        grad=rosenbrock_grad,
        x0=np.array([-1.5, 2.0]),
        n_iterations=2000,
        true_minimum=0.0,
        name="Rosenbrock f(x,y) = 100(y-x^2)^2 + (1-x)^2",
    )

    # --- 2. Rastrigin Function ---
    print("\n2. Rastrigin Function Optimization")
    print("-" * 60)
    compare_optimizers(
        f=rastrigin,
        grad=rastrigin_grad,
        x0=np.array([3.0, 3.0]),
        n_iterations=5000,
        true_minimum=0.0,
        name="Rastrigin f(x) = 20 + x^2 - 10cos(2*pi*x)",
    )

    # --- 3. Sphere Function ---
    print("\n3. Sphere Function Optimization (using numerical gradients)")
    print("-" * 60)
    adam = AdamOptimizer(learning_rate=0.1)
    result = adam.optimize(sphere, np.array([5.0, -3.0]), grad=None, n_iterations=500, verbose=True)
    print(f"\n  Final x = {result['x'].round(8)}")
    print(f"  Final f(x) = {result['f']:.2e}")
    print(f"  Converged: {result['converged']}")

    # --- 4. Learning Rate Schedulers ---
    print("\n" + "=" * 60)
    print("4. Learning Rate Schedulers Comparison")
    print("=" * 60)

    n_iters = 500
    schedulers = [
        ("Step Decay", StepDecay(0.1, drop_factor=0.5, step_size=100)),
        ("Exponential Decay", ExponentialDecay(0.1, decay_rate=0.01)),
        ("Cosine Annealing", CosineAnnealing(0.1, eta_min=1e-5, T_max=n_iters)),
    ]

    print(f"\n  Initial LR = 0.1, Iterations = {n_iters}\n")
    for sched_name, scheduler in schedulers:
        lrs = [scheduler.get_lr(i) for i in range(n_iters)]
        print(f"  {sched_name}:")
        print(f"    LR at iter 0:   {lrs[0]:.6f}")
        print(f"    LR at iter 100: {lrs[100]:.6f}")
        print(f"    LR at iter 250: {lrs[250]:.6f}")
        print(f"    LR at iter 499: {lrs[-1]:.6f}")

    # --- 5. Optimization with LR Scheduling ---
    print("\n" + "=" * 60)
    print("5. Adam with Exponential LR Schedule on Rosenbrock")
    print("=" * 60)

    x = np.array([-1.5, 2.0], dtype=np.float64)
    adam2 = AdamOptimizer(learning_rate=0.01)
    scheduler = ExponentialDecay(initial_lr=0.01, decay_rate=0.002)

    history_scheduled = []
    for i in range(2000):
        current_lr = scheduler.get_lr(i)
        adam2.learning_rate = current_lr

        result = adam2.optimize(
            rosenbrock, x, grad=rosenbrock_grad, n_iterations=1, verbose=False
        )
        x = result["x"]
        history_scheduled.append(result["f"])

    print(f"\n  Final x = {x.round(6)}")
    print(f"  Final f(x) = {rosenbrock(x):.8e}")
    print(f"  Final LR = {current_lr:.8f}")

    # --- 6. Beale Function ---
    print("\n" + "=" * 60)
    print("6. Beale Function Optimization")
    print("=" * 60)
    adam3 = AdamOptimizer(learning_rate=0.05)
    result = adam3.optimize(
        beale,
        np.array([0.0, 0.0]),
        grad=None,  # Use numerical gradient
        n_iterations=5000,
        verbose=False,
    )
    print(f"  Beale function minimum: x = [3, 0.5], f = 0")
    print(f"  Adam result: x = {result['x'].round(4)}, f = {result['f']:.6e}")

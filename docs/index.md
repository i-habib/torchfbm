# torchfbm: Fractional Brownian Motion in PyTorch

**torchfbm** is a PyTorch library for the efficient generation, analysis, and modeling of Fractional Brownian Motion (fBm) and Fractional Gaussian Noise (fGn). It provides GPU-accelerated solvers, differentiable estimators, and neural network layers designed for stochastic modeling and anomalous diffusion research.

The library bridges the gap between classical stochastic calculus and modern deep learning, enabling end-to-end training of systems governed by fractional noise.

---

## Key Capabilities

*   **Efficient Generation:** Implements the Davies-Harte algorithm (Exact FFT-based simulation) with $O(N \log N)$ complexity, fully vectorized for batched GPU generation, as well as Cholesky Decomposition for exact covariance matrix factorization.
*   **Differentiable Estimators:** Includes Detrended Fluctuation Analysis (DFA) and spectral estimators that allow gradients to propagate through the Hurst exponent estimation.
*   **Neural SDE Solvers:** Provides Euler-Maruyama solvers for Fractional Stochastic Differential Equations (fSDEs) with learnable drift, diffusion, and Hurst parameters.
*   **Deep Learning Integration:** Features `FBMNoisyLinear` layers for structured exploration in Reinforcement Learning and `FractionalKernel` for Gaussian Processes.

## Installation

```bash
pip install torchfbm
```

## Quick Start

### Generating Correlated Noise

`torchfbm` serves as a drop-in replacement for standard Gaussian noise when temporal correlation is required.

```python
import torch
from torchfbm import generate_davies_harte

# Generate a batch of 100 paths, each with 10,000 steps
# H=0.7 implies persistent memory (long-range dependence)
noise = generate_davies_harte(
    n=10000, 
    H=0.7, 
    size=(100,), 
    device='cuda'
)

print(f"Shape: {noise.shape}")  # torch.Size([100, 10000])
print(f"Device: {noise.device}") # cuda:0
```

### Solving a Fractional SDE

Solve equations of the form $dX_t = \mu(X_t)dt + \sigma(X_t)dB^H_t$.

```python
from torch import nn
from torchfbm.sde import NeuralFSDE

# Define drift and diffusion networks
drift_net = nn.Linear(1, 1)
diffusion_net = nn.Linear(1, 1)

# Initialize solver with learnable Hurst parameter
fsde = NeuralFSDE(
    state_size=1, 
    drift_net=drift_net, 
    diffusion_net=diffusion_net,
    learnable_H=True,
    H_init=0.7
)

# Integrate trajectory
x0 = torch.zeros(32, 1).cuda()
trajectory = fsde(x0, n_steps=1000)
```

## Mathematical Foundation

Fractional Brownian Motion $B_H(t)$ is a continuous-time Gaussian process characterized by the Hurst exponent $H \in (0, 1)$. Its covariance function is given by Mandelbrot & Van Ness (1968):

$$ E[B_H(t)B_H(s)] = \frac{1}{2} \left(|t|^{2H} + |s|^{2H} - |t-s|^{2H} \right) $$

*   **$H < 0.5$**: Anti-persistent (rough volatility, mean-reverting).
*   **$H = 0.5$**: Standard Brownian Motion (uncorrelated increments).
*   **$H > 0.5$**: Persistent (trending, long memory).

## Citation

If you use `torchfbm` in your research, please cite:

```bibtex
@software{torchfbm2026,
  author = {Ivan Habib},
  title = {TorchFBM: High-performance Fractional Brownian Motion toolkit for PyTorch},
  year = {2026},
  url = {https://github.com/i-habib/torchfbm}
}
```

# `torchfbm`
### Differentiable Fractional Brownian Motion & Rough Volatility for PyTorch

[![Tests](https://github.com/i-habib/torchfbm/actions/workflows/tests.yml/badge.svg)](https://github.com/i-habib/torchfbm/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://i-habib.github.io/torchfbm/)

`torchfbm` generates and analyzes fractional Brownian motion (fBm) and fractional Gaussian noise (fGn) in PyTorch. The generators run on CPU or GPU and are differentiable, so the Hurst parameter and the paths can sit inside a training loop.

## What's included

**Generators**
- `fbm(..., method='davies_harte')`: FFT-based circulant embedding, O(N log N).
- `fbm(..., method='cholesky')`: exact Cholesky factorization, O(N³), for checking the fast method.
- `CachedFGNGenerator`: one new sample at a time by incremental Cholesky, O(N²) per step.

**Processes**
- `geometric_fbm`, `fractional_ou_process`, `multifractal_random_walk`
- `reflected_fbm`, `fractional_brownian_bridge`
- `fractional_diff` (fractional differencing)

**Estimation**
- `estimate_hurst` (aggregated variance, differentiable)
- `dfa` (detrended fluctuation analysis)

**Neural network pieces**
- `FBMNoisyLinear`: a noisy linear layer whose noise is fGn instead of white noise
- `FractionalPositionalEmbedding`
- `SpectralConsistencyLoss`: penalizes deviation from a 1/f^β power spectrum
- `get_hurst_schedule`: a schedule for H across diffusion steps
- `NeuralFSDE`: a neural SDE driven by fBm with a learnable H

## Install

From PyPI:
```bash
pip install torchfbm
```

For development:
```bash
git clone https://github.com/i-habib/torchfbm.git
cd torchfbm
pip install -e .
```

## Quick Usage

### 1. Generate Rough Paths (Batch)
Generate fractional noise on CUDA using the fast Davies-Harte method.

```python
import torch
from torchfbm import fbm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 4 paths of length 1024 with H=0.7 (positively correlated increments)
path = fbm(n=1024, H=0.7, size=(4,), method='davies_harte', device=device)
```

### 2. Real-Time Streaming (Online)
Use `CachedFGNGenerator` to draw one sample at a time.

```python
from torchfbm.online import CachedFGNGenerator

stream = CachedFGNGenerator(H=0.3, device=device)  # H < 0.5: negatively correlated increments

for i in range(100):
    val = stream.step()
    print(f"Tick {i}: {val.item():.4f}")
```

### 3. Noisy layers
Replace standard `nn.Linear` with `FBMNoisyLinear`.

```python
from torchfbm import FBMNoisyLinear

# Initialize layer with H=0.5 (Standard)
layer = FBMNoisyLinear(32, 10, H=0.5, device=device)

# change H and redraw the noise
layer.H = 0.2
layer.refresh_noise_stream()
y = layer(torch.randn(8, 32, device=device))
```

### 4. Hurst schedules for diffusion
Vary H across the reverse process.

```python
from torchfbm.schedulers import get_hurst_schedule

hs = get_hurst_schedule(n_steps=1000, start_H=0.3, end_H=0.7, type='cosine')

for t in reversed(range(1000)):
    current_H = hs[t]
    # Use current_H for sampling noise...
```

### 5. Processes

```python
from torchfbm import geometric_fbm, fractional_ou_process, multifractal_random_walk

s = geometric_fbm(n=1000, H=0.7, mu=0.05, sigma=0.2, s0=100.0, device=device)

mrw = multifractal_random_walk(n=1000, H=0.3, lambda_sq=0.02, device=device)
```

## Analysis Tools

```python
from torchfbm import estimate_hurst, fractional_diff, dfa

# aggregated-variance Hurst estimate (differentiable)
H_est = estimate_hurst(path.unsqueeze(0), min_lag=4, max_lag=64)

# detrended fluctuation analysis
alpha = dfa(path, scales=None, order=1, return_alpha=True)

# fractional differencing
stationary_ts = fractional_diff(path, d=0.4)
```

## Notes

- Use `method='davies_harte'` for long paths and `method='cholesky'` to check results exactly.
- H is clamped to [0.01, 0.99].
- Tests: `pytest torchfbm/tests` (511 tests).
- MIT licensed.

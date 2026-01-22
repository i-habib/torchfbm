
---
title: 'torchfbm: A PyTorch Library for Differentiable Fractional Brownian Motion'
tags:
  - Python
  - PyTorch
  - Stochastic Processes
  - Fractional Brownian Motion
  - Deep Learning
  - Anomalous Diffusion
authors:
  - name: Ivan Habib
    orcid: 0009-0002-7812-0371
    
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 22 January 2026
bibliography: paper.bib
---

# Summary

`torchfbm` is a Python library built upon PyTorch that facilitates the simulation, analysis, and modeling of Fractional Brownian Motion (fBm) and Fractional Gaussian Noise (fGn). The library provides GPU-accelerated implementations of the Davies-Harte algorithm for exact simulation [@davies1987tests], alongside differentiable modules for integrating fractional noise into neural network architectures. `torchfbm` extends standard stochastic differential equation (SDE) solvers to the fractional regime, enabling the modeling of systems exhibiting anomalous diffusion and long-range dependence.

# Statement of Need

Fractional Brownian Motion is a fundamental stochastic process used to model phenomena with long-term memory in fields ranging from quantitative finance [@rogers1997arbitrage] to biophysics and hydrology [@mandelbrot1968fractional]. While Python packages such as `fbm` exist, they typically rely on NumPy, restricting computations to the CPU and preventing gradient propagation.

As deep learning intersects with stochastic modeling—specifically in Physics-Informed Neural Networks (PINNs) and Neural SDEs—there is a critical need for a library that treats $H$ (the Hurst exponent) not just as a static hyperparameter, but as a differentiable tensor. `torchfbm` addresses this by providing:

1.  **Computational Efficiency:** Leveraging PyTorch's FFT implementation to execute the Davies-Harte algorithm on GPUs, reducing generation time for large batches of long path lengths ($N > 10^4$).
2.  **Differentiability:** Enabling end-to-end optimization of the Hurst exponent via spectral estimators and Detrended Fluctuation Analysis (DFA) [@peng1994mosaic].
3.  **Modular Abstractions:** Offering `nn.Module` wrappers for Noisy Nets [@fortunato2017noisy] and Neural fSDEs, facilitating the exploration of non-Markovian dynamics in Reinforcement Learning and Generative Modeling.

# Mathematics and Implementation

The core of `torchfbm` relies on the spectral embedding of the autocovariance sequence of fGn into a circulant matrix. For a process with Hurst exponent $H$, the autocovariance $\gamma(k)$ is defined as:

$$ \gamma(k) = \frac{1}{2} \left(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H} \right) $$

By computing the eigenvalues of the circulant matrix via Fast Fourier Transform (FFT), `torchfbm` generates exact fGn samples with $O(N \log N)$ complexity.

The library also implements the Fractional Ornstein-Uhlenbeck process and Geometric fBm, serving as foundational building blocks for quantitative finance research.

# Research Applications

`torchfbm` is designed to support research in:

*   **Reinforcement Learning:** Using `FBMNoisyLinear` and `FBMActionNoise` to introduce temporally correlated exploration noise, which has been shown to improve convergence in sparse-reward environments.
*   **Generative Modeling:** Training Neural SDEs to approximate rough volatility surfaces or time-series data with specific spectral scaling properties.
*   **Signal Processing:** Utilizing GPU-accelerated DFA to estimate scaling exponents in high-frequency data streams.

# References
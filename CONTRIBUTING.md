# Contributing to TorchFBM

Thank you for your interest in contributing! This guide will help you get started.

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/i-habib/torchfbm.git
cd torchfbm
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**
```bash
pip install -e ".[dev]"
```

Or install from requirements:
```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

## Running Tests

Run the full test suite:
```bash
pytest torchfbm/tests/ -v
```

Run with coverage:
```bash
pytest torchfbm/tests/ --cov=torchfbm --cov-report=html
open htmlcov/index.html  # View coverage report
```

Run a specific test file:
```bash
pytest torchfbm/tests/test_generators.py -v
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines
- Use **Google-style docstrings** for all public functions and classes
- Include **type hints** for function parameters and return values
- Add **LaTeX formulas** in docstrings for mathematical functions (using `$$...$$`)
- Keep lines under 88 characters (Black formatter compatible)

### Docstring Example

```python
def my_function(x: torch.Tensor, H: float) -> torch.Tensor:
    """Short description of the function.

    Longer description with mathematical details.

    The formula is:

    $$f(x) = x^{2H}$$

    Based on Author & Author (Year).

    Args:
        x: Input tensor of shape ``(batch, n)``.
        H: Hurst parameter in (0, 1).

    Returns:
        Output tensor with same shape as input.

    Raises:
        ValueError: If H is not in valid range.

    Example:
        >>> result = my_function(torch.randn(10, 100), H=0.7)
    """
```

## Pull Request Process

1. **Fork** the repository and create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** with clear, atomic commits

3. **Add tests** for any new functionality

4. **Ensure all tests pass:**
```bash
pytest torchfbm/tests/ -v
```

5. **Update documentation** if you changed public APIs

6. **Submit a pull request** with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable

## Documentation

Build docs locally:
```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]"
mkdocs serve
```

Then visit http://127.0.0.1:8000

Documentation is auto-deployed to GitHub Pages on push to `main`.

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/i-habib/torchfbm/issues) to report bugs or request features.

**For bug reports, include:**
- Python version (`python --version`)
- PyTorch version (`python -c "import torch; print(torch.__version__)"`)
- Operating system
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

**For feature requests, include:**
- Use case / motivation
- Proposed API (if applicable)
- References to relevant papers or implementations

## Questions?

Feel free to open a GitHub issue for questions about the codebase or contribution process.

Thank you for helping improve TorchFBM! 🚀

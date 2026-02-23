# Contributing to InfraWatch

Thank you for your interest in contributing to InfraWatch. This document provides
guidelines and information for contributors.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/cwccie/infrawatch.git
cd infrawatch

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all extras
pip install -e ".[dev,ml]"

# Run tests
pytest tests/ -v

# Run linter
ruff check src/ tests/

# Run type checker
mypy src/infrawatch/ --ignore-missing-imports
```

## Project Structure

```
src/infrawatch/
├── collect/      # Metric collection (Prometheus, SNMP, StatsD, files)
├── preprocess/   # Time series preprocessing pipeline
├── models/       # Anomaly detection models
├── detect/       # Detection pipeline and severity classification
├── maintenance/  # Maintenance window management
├── alert/        # Alert engine and notification backends
├── forecast/     # Capacity forecasting
├── api/          # Flask REST API
├── dashboard/    # Web dashboard
└── cli.py        # CLI interface
```

## Guidelines

### Code Style

- Follow PEP 8 (enforced by `ruff`)
- Use type hints for all public function signatures
- Write docstrings for all public modules, classes, and functions
- Keep functions focused and under 50 lines where practical

### Testing

- All new features must include tests
- Maintain test coverage above 80%
- Use the fixtures in `tests/conftest.py` where appropriate
- Test edge cases: empty data, NaN values, single-point series

### Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Use imperative mood: "Fix bug" not "Fixes bug"
- Reference issues where applicable: "Fix anomaly scoring (#42)"

### Pull Requests

1. Fork the repository and create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass: `pytest tests/ -v`
4. Ensure linting passes: `ruff check src/ tests/`
5. Update documentation if applicable
6. Submit a pull request with a clear description

## Architecture Decisions

- **NumPy/SciPy core**: All statistical and ML models use NumPy/SciPy to
  minimize dependencies. scikit-learn is optional.
- **Foundation models are optional**: The core system works without GPU or
  large model downloads. Foundation model support is behind an optional
  dependency.
- **Zero-config philosophy**: The system should work with sensible defaults.
  Advanced configuration is available but never required.
- **Maintenance-window awareness**: Alert suppression during maintenance is
  a first-class feature, not an afterthought.

## Reporting Issues

- Use GitHub Issues for bugs and feature requests
- Include Python version, OS, and InfraWatch version
- For bugs, include a minimal reproduction example
- For features, describe the use case and expected behavior

## License

By contributing, you agree that your contributions will be licensed under
the MIT License.

# Contributing to DeepSDF

Thank you for your interest in contributing to DeepSDF! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DeepSDF.git
   cd DeepSDF
   ```

3. Install uv if you haven't already:
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

4. Install in development mode:
   ```bash
   uv sync --all-extras
   ```

5. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Code Quality Standards

We maintain high code quality standards. All contributions must:

### 1. Pass Code Formatting

Format your code with Black:
```bash
uv run black deepsdf/ examples/ tests/
```

### 2. Pass Linting

Ensure no linting errors:
```bash
uv run flake8 deepsdf/ examples/ tests/
```

### 3. Pass Type Checking

Verify type hints:
```bash
uv run mypy deepsdf/ --ignore-missing-imports
```

### 4. Pass Tests

All tests must pass:
```bash
uv run pytest tests/ -v
```

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use descriptive variable names

### Documentation

- Add docstrings to all public functions/classes
- Use Google-style docstrings
- Include type information in docstrings
- Add examples where helpful

Example:
```python
def my_function(arg1: int, arg2: str) -> bool:
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Example:
        >>> my_function(1, "test")
        True
    """
    pass
```

### Type Hints

All functions should have type hints:
```python
def process_data(
    data: np.ndarray,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names: `test_<functionality>_<condition>`
- Test both success and failure cases
- Aim for >80% code coverage

Example:
```python
def test_decoder_forward_pass() -> None:
    """Test decoder forward pass with valid inputs."""
    model = DeepSDFDecoder(latent_size=256)
    latent = torch.randn(4, 256)
    xyz = torch.randn(4, 1000, 3)
    
    output = model(latent, xyz)
    
    assert output.shape == (4, 1000, 1)
    assert not torch.isnan(output).any()
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=deepsdf

# Run specific test
uv run pytest tests/test_decoder.py::test_decoder_forward
```

## Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Quality Checks**
   ```bash
   uv run black deepsdf/ examples/ tests/
   uv run flake8 deepsdf/ examples/ tests/
   uv run mypy deepsdf/ --ignore-missing-imports
   uv run pytest tests/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```
   
   Use descriptive commit messages:
   - `feat: Add new feature`
   - `fix: Fix bug in module`
   - `docs: Update documentation`
   - `test: Add tests for feature`
   - `refactor: Refactor code`

5. **Push to Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Provide clear description of changes
   - Reference any related issues
   - Ensure CI passes

## What to Contribute

### Good First Issues

- Bug fixes
- Documentation improvements
- Additional tests
- Example scripts
- Performance optimizations

### Feature Requests

Before implementing a major feature:
1. Open an issue to discuss the feature
2. Get feedback from maintainers
3. Create a design doc if needed
4. Implement with tests and docs

## Code Review Process

All submissions require review. We look for:

- **Correctness**: Does it work as intended?
- **Testing**: Are there adequate tests?
- **Documentation**: Is it well-documented?
- **Style**: Does it follow our guidelines?
- **Design**: Is it well-designed and maintainable?

## Reporting Bugs

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal example to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, etc.

Example:
```markdown
**Bug**: Model crashes during training

**Steps to Reproduce**:
\`\`\`python
model = DeepSDFDecoder()
# ... code to reproduce
\`\`\`

**Expected**: Model trains successfully
**Actual**: RuntimeError: ...

**Environment**: Python 3.9, Ubuntu 20.04, PyTorch 1.12
```

## Questions?

- Open an issue for questions
- Check existing issues first
- Be respectful and constructive

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to DeepSDF! 🎉

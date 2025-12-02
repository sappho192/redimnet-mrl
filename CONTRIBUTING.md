# Contributing to ReDimNet-MRL

Thank you for your interest in contributing to ReDimNet-MRL! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## How to Contribute

### Reporting Bugs

**Before submitting a bug report**:
- Check existing issues to avoid duplicates
- Test with the latest version
- Collect relevant information (OS, PyTorch version, GPU, error messages)

**Bug report should include**:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- System information
- Code snippets or error logs

### Suggesting Enhancements

**Feature requests should include**:
- Clear description of the feature
- Motivation and use case
- Examples of how it would work
- Any implementation ideas

### Pull Requests

**Before submitting**:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test your changes
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

**PR guidelines**:
- One feature/fix per PR
- Follow existing code style
- Add tests if applicable
- Update documentation
- Keep commits clean and focused

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/redimnet-mrl.git
cd redimnet-mrl

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests (when available)
pytest tests/
```

## Code Style

**Python**:
- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions/classes
- Keep lines under 100 characters (120 max)

**Example**:
```python
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100,
) -> Dict[str, float]:
    """
    Train the MRL model.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer instance
        num_epochs: Number of training epochs

    Returns:
        Dictionary with training metrics
    """
    # Implementation here
    pass
```

## Testing

**When adding new features**:
- Add unit tests
- Test with different configurations
- Verify GPU memory usage
- Check compatibility with pretrained models

**Test checklist**:
- [ ] Code runs without errors
- [ ] Memory usage is reasonable
- [ ] Performance is not degraded
- [ ] Documentation is updated
- [ ] Examples work correctly

## Documentation

**Update documentation when**:
- Adding new features
- Changing APIs
- Fixing bugs that affect usage
- Adding new configuration options

**Documentation locations**:
- `README.md`: Main overview
- `*.md` files: Specific guides
- Docstrings: Code documentation
- `CHANGELOG.md`: Version changes

## Commit Messages

**Format**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (no logic change)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:
```bash
feat(model): add support for b7 variant
fix(training): correct gradient accumulation logic
docs(readme): update installation instructions
```

## Areas for Contribution

### High Priority

1. **Evaluation Tools**
   - Multi-dimension EER evaluation
   - Benchmark suite for VoxCeleb1
   - Performance visualization

2. **Documentation**
   - More usage examples
   - Video tutorials
   - API reference

3. **Testing**
   - Unit tests for all modules
   - Integration tests
   - CI/CD setup

### Medium Priority

4. **Optimization**
   - Distributed training support
   - Memory optimization
   - Inference speedup

5. **Features**
   - Additional datasets support
   - LoRA integration
   - Model compression

6. **Tools**
   - Inference server
   - Model export (ONNX, TorchScript)
   - Web demo

### Research Extensions

7. **Novel Features**
   - Cross-model distillation
   - Progressive MRL training
   - Multi-task learning
   - Extreme low dimensions (8D, 16D, 32D)

## Questions?

- Open a GitHub issue for questions
- Check existing documentation
- Review closed issues for similar questions

## Recognition

Contributors will be recognized in:
- `CHANGELOG.md`
- GitHub contributors page
- Release notes

Thank you for contributing! 🎉

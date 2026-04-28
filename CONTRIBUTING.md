# Contributing to CSA (Compressed Speculative Attention)

Thank you for your interest in contributing to CSA! This document provides guidelines for contributing.

## 🚀 Current Project Status (April 2026)

**✅ Functional Components:**
- KV Cache Compression: 5-50x reduction (VERIFIED)
- FP8 Quantization: Working (MSE: 0.001331)
- Custom Attention Layer: `CompressedAttention` (IMPLEMENTED)
- Multi-Model Support: GPT-2, LLaMA, OPT (WORKING)
- 52 tests passing

**⚠️ In Development:**
- Speedup Verification: End-to-end benchmarks pending
- SSD Speculation: Framework ready, integration pending
- Background Recovery: Framework ready

## 🎯 How to Contribute

### 1. Report Bugs
Found a bug? Please include:
- Python version and OS
- PyTorch version
- Error message and traceback
- Steps to reproduce

### 2. Suggest Features
We welcome new ideas! Especially:
- New model support (Mistral, Gemma, etc.)
- Optimized CUDA kernels
- Additional quantization schemes
- Performance optimizations

### 3. Submit Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass: `python -m pytest tests/ -v`
6. Update documentation
7. Submit PR with clear description

### 4. Improve Documentation
- Fix typos, clarify explanations
- Add examples for new features
- Update benchmarks with verified numbers

## 🔧 Development Setup

```bash
git clone https://github.com/kishoretvk/DevClaw.git
cd DevClaw
pip install -e .
pip install pytest  # For running tests
```

## 📊 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_csa_comprehensive.py -v

# Run with coverage
python -m pytest --cov=csa tests/
```

## 📚 Current Focus Areas

### 1. Speedup Verification (High Priority)
- Run end-to-end benchmarks with `CompressedAttention`
- Measure actual speedup vs baseline
- Update documentation with verified numbers

### 2. SSD Speculation Integration
- Complete `csa/speculation/ssd.py` integration
- Test with draft models
- Verify 2-3x speedup claims

### 3. New Model Support
- Add support for Mistral, Gemma, Phi, etc.
- Update `csa/attention/patcher.py` with new model types
- Add tests for new models

### 4. Performance Optimization
- Optimize CUDA kernels for attention
- Reduce overhead in compression/decompression
- Profile and benchmark on GPU

## 📖 Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all public methods
- Keep functions focused and modular

## 🤝 Questions?

Feel free to open an issue with questions about contributing!

---

**Current Maintainer**: Krishna (TheExploreEcho)  
**Last Updated**: April 27, 2026  
**Status**: Functional proof-of-concept with verified compression & quantization
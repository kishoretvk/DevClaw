# PyPI Publishing Setup Guide

This guide explains how to set up automated PyPI publishing for the CSA package using GitHub Actions.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **API Token**: Generate a token at https://pypi.org/manage/account/token/
3. **GitHub Repository**: Admin access to the DevClaw repository

## Setup Steps

### 1. Create PyPI API Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Set scope to "Entire account (all projects)"
4. Give it a descriptive name like "CSA GitHub Actions"
5. Copy the generated token (you won't see it again!)

### 2. Add Token to GitHub Secrets

1. Go to your repository: https://github.com/kishoretvk/DevClaw
2. Click "Settings" tab
3. Click "Secrets and variables" → "Actions" in left sidebar
4. Click "New repository secret"
5. Name: `PYPI_API_TOKEN`
6. Value: Paste your PyPI API token
7. Click "Add secret"

### 3. Create GitHub Environment (Optional but Recommended)

For better security, create a "release" environment:

1. Go to repository Settings → Environments
2. Click "New environment"
3. Name: `release`
4. Add protection rule: Require approval for production deployments
5. Add the `PYPI_API_TOKEN` secret to this environment

### 4. Test the Setup

Create a test release to verify everything works:

1. Go to repository main page
2. Click "Releases" in right sidebar
3. Click "Create a new release"
4. Tag version: `v0.1.0-test` (or similar test version)
5. Title: "Test Release v0.1.0"
6. Mark as pre-release
7. Click "Publish release"

This will trigger the PyPI publishing workflow.

## How It Works

### Automatic Publishing
- **Trigger**: When you create a GitHub release (tag like `v1.0.0`)
- **Process**:
  1. Runs tests on multiple Python versions (3.8, 3.9, 3.10, 3.11, 3.12)
  2. Builds the package (wheel + source distribution)
  3. Publishes to PyPI using trusted publishing (secure)

### Manual Publishing
You can also manually trigger publishing:
1. Go to repository Actions tab
2. Click "Publish to PyPI" workflow
3. Click "Run workflow"
4. Optionally specify a version

## Testing Workflow

### On Pull Requests
- **Trigger**: PRs to main/develop branches
- **Tests**: Linting, unit tests, package build verification
- **Purpose**: Ensure code quality before merging

### On Pushes
- **Trigger**: Pushes to main/develop branches
- **Tests**: Full test suite
- **Purpose**: Catch regressions early

## Troubleshooting

### Publishing Fails
**Check:**
1. PyPI API token is correct and has "Entire account" scope
2. Token is added to GitHub secrets as `PYPI_API_TOKEN`
3. Package version hasn't been published before
4. Package name `csa` is available on PyPI

### Tests Fail
**Check:**
1. All dependencies are properly specified in `pyproject.toml`
2. Tests pass locally: `pytest tests/`
3. Package builds locally: `python -m build`

### Permission Issues
**Check:**
1. Repository has GitHub Actions enabled
2. `GITHUB_TOKEN` has appropriate permissions
3. PyPI token has upload permissions

## Security Best Practices

- ✅ **Trusted Publishing**: Uses PyPI's trusted publishing (no plain text tokens)
- ✅ **Environment Protection**: Release environment requires approval
- ✅ **Minimal Permissions**: Workflows have minimal required permissions
- ✅ **Token Rotation**: Regularly rotate PyPI API tokens

## Release Process

1. **Update version** in `pyproject.toml`
2. **Update changelog** if you have one
3. **Commit changes** to main branch
4. **Create GitHub release** with version tag (e.g., `v1.0.0`)
5. **Monitor Actions tab** for publishing progress
6. **Verify on PyPI**: https://pypi.org/project/csa/

## Monitoring

Check publishing status:
- **GitHub Actions**: Repository → Actions → "Publish to PyPI"
- **PyPI**: https://pypi.org/project/csa/
- **Test Installation**: `pip install csa --dry-run`

---

**Ready for automated PyPI publishing!** 🚀
"""Dataset, preflight, training, and validation tools.

The runtime shares only the lightweight XML validation module. Torch and the
training modules remain lazily imported and outside the Windows edge package.
"""

"""Offline-only dataset, training, and validation tools.

Production modules never import this package, keeping Torch and training-only
dependencies out of the Windows edge runtime.
"""

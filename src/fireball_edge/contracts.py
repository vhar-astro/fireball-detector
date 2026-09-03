"""Versioned contracts shared by edge runtime and offline training.

Changing any of these values intentionally makes previously generated
manifests, caches, model packages, and results incompatible.  That is a
feature: v2 deliberately has different source semantics from the retired
change-map pipeline.
"""

from __future__ import annotations


SCHEMA_VERSION = 2
CANDIDATE_EXTRACTOR = "avi-diff-stack-v2"


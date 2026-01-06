"""
Pytest configuration for korg tests.

This file ensures JAX x64 mode is enabled before any tests run.
"""

import os

# Set x64 mode via environment variable BEFORE any imports
os.environ["JAX_ENABLE_X64"] = "true"

# Now import korg which will also set x64 mode
import korg  # noqa: E402, F401

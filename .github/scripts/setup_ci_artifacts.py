#!/usr/bin/env python3
"""
Setup script for CI environments.

Creates placeholder artifacts so that tests can import modules
without downloading large data files.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from korg.artifacts import create_placeholder_artifact, list_artifacts


def main():
    """Create placeholder artifacts for CI."""
    print("Setting up placeholder artifacts for CI...")

    # Set CI flag
    os.environ['CI'] = '1'

    # Create placeholders for all MARCS artifacts
    marcs_artifacts = [
        'SDSS_MARCS_atmospheres_v2',
        'MARCS_metal_poor_atmospheres',
        'resampled_cool_dwarf_atmospheres'
    ]

    for artifact_name in marcs_artifacts:
        try:
            print(f"\nCreating placeholder for {artifact_name}...")
            artifact_dir = create_placeholder_artifact(artifact_name)
            print(f"  ✓ Created at {artifact_dir}")
        except Exception as e:
            print(f"  ✗ Failed: {e}", file=sys.stderr)
            return 1

    # List status
    print("\nArtifact status:")
    status = list_artifacts()
    for name, available in status.items():
        symbol = "✓" if available else "✗"
        print(f"  {symbol} {name}")

    print("\nPlaceholder setup complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

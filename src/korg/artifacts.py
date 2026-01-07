"""
Artifact download system for Korg data files.

This module provides functionality similar to Julia's Pkg.Artifacts system,
automatically downloading large data files from AWS S3 when needed.

The artifacts are downloaded to ~/.korg/ by default, matching Julia's behavior.
"""

import os
import hashlib
import tarfile
import warnings
import urllib.request
from pathlib import Path
from typing import Optional, Dict


# Default data directory (can be overridden with KORG_DATA_DIR environment variable)
def get_korg_data_dir() -> Path:
    """
    Get the Korg data directory path.

    Returns
    -------
    Path
        Path to ~/.korg or directory specified by KORG_DATA_DIR
    """
    data_dir = os.environ.get('KORG_DATA_DIR')
    if data_dir:
        return Path(data_dir)
    return Path.home() / '.korg'


# Artifact registry matching Artifacts.toml
ARTIFACTS = {
    'SDSS_MARCS_atmospheres_v2': {
        'url': 'https://korg-data.s3.amazonaws.com/SDSS_MARCS_atmospheres_v2.tar.gz',
        'sha256': '98a2d96805d255539e47ef3ae8671bcf776ca80cc4a50e09a47100c8477d80be',
        'git_tree_sha1': '0e9cca0d99f7ad79aa966aa28c7be962071893e2',
        'extract_dir': 'SDSS_MARCS_atmospheres',
        'files': ['SDSS_MARCS_atmospheres.h5']
    },
    'MARCS_metal_poor_atmospheres': {
        'url': 'https://korg-data.s3.amazonaws.com/MARCS_metal_poor_atmospheres.tar.gz',
        'sha256': 'd08dce5f1bb98e2f8b74db563ad47182722ba354d6cf8690aacac28bbd1f64f0',
        'git_tree_sha1': '7ccbdaf1a3709772953d75b0888e6d5ec359d90b',
        'extract_dir': 'MARCS_metal_poor_atmospheres',
        'files': ['MARCS_metal_poor_atmospheres.h5']
    },
    'resampled_cool_dwarf_atmospheres': {
        'url': 'https://korg-data.s3.amazonaws.com/resampled_cool_dwarf_atmospheres.tar.gz',
        'sha256': '385133c3b06f0a632ec05e3e58feaa7587b673c4a805eea07b9d0aa33386e940',
        'git_tree_sha1': 'cd6250126e449334f340641a545c2ab5c3d1bf7c',
        'extract_dir': 'resampled_cool_dwarf_atmospheres',
        'files': ['resampled_cool_dwarf_atmospheres.h5']
    },
    'Heiter_2021_GES_linelist': {
        'url': 'https://korg-data.s3.amazonaws.com/Heiter_el_al_2021_2024_06_17.tar.gz',
        'sha256': 'f6eedce5e5a1a16428875b4a0ad94bc9fd205098d285ac7a5598c70091f3446e',
        'git_tree_sha1': '449099fde28068282516612ed15fdda5a71bc4ec',
        'extract_dir': 'Heiter_el_al_2021',
        'files': ['linelist.vald']
    }
}


def compute_sha256(filepath: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Parameters
    ----------
    filepath : Path
        Path to file

    Returns
    -------
    str
        Hexadecimal SHA256 hash
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def is_placeholder_file(filepath: Path) -> bool:
    """
    Check if a file is a placeholder (0 or very small size).

    Placeholder files are used in CI environments where actual data
    downloads are skipped.

    Parameters
    ----------
    filepath : Path
        Path to check

    Returns
    -------
    bool
        True if file is a placeholder
    """
    if not filepath.exists():
        return False

    # Consider files < 1 KB as placeholders
    return filepath.stat().st_size < 1024


def download_artifact(
    artifact_name: str,
    force: bool = False,
    show_progress: bool = True
) -> Path:
    """
    Download and extract an artifact if not already present.

    This function mimics Julia's lazy artifact system. It downloads the
    artifact from AWS S3, verifies the SHA256 hash, and extracts it to
    the Korg data directory.

    Parameters
    ----------
    artifact_name : str
        Name of artifact (must be in ARTIFACTS registry)
    force : bool, optional
        Force re-download even if artifact exists (default: False)
    show_progress : bool, optional
        Show download progress (default: True)

    Returns
    -------
    Path
        Path to extracted artifact directory

    Raises
    ------
    ValueError
        If artifact_name is not in registry
    RuntimeError
        If download fails or hash verification fails

    Examples
    --------
    >>> path = download_artifact('SDSS_MARCS_atmospheres_v2')
    >>> print(path / 'SDSS_MARCS_atmospheres' / 'SDSS_MARCS_atmospheres.h5')
    """
    if artifact_name not in ARTIFACTS:
        raise ValueError(f"Unknown artifact: {artifact_name}. "
                        f"Available: {list(ARTIFACTS.keys())}")

    artifact = ARTIFACTS[artifact_name]
    data_dir = get_korg_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use git-tree-sha1 as directory name (matching Julia's Pkg.Artifacts)
    artifact_dir = data_dir / artifact['git_tree_sha1']
    extract_dir = artifact_dir / artifact['extract_dir']

    # Check if artifact already exists and is valid
    if not force and extract_dir.exists():
        # Check if it's a placeholder file (for CI)
        main_file = extract_dir / artifact['files'][0]
        if is_placeholder_file(main_file):
            if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
                warnings.warn(
                    f"Using placeholder for {artifact_name} in CI environment. "
                    "Interpolation will not work.",
                    UserWarning
                )
                return artifact_dir

        # If all expected files exist, assume it's valid
        all_exist = all((extract_dir / f).exists() for f in artifact['files'])
        if all_exist:
            return artifact_dir

    # Download to temporary file
    tarball_path = data_dir / f"{artifact_name}.tar.gz"

    if show_progress:
        print(f"Downloading {artifact_name} from AWS S3...")
        print(f"URL: {artifact['url']}")

    try:
        # Download with progress reporting
        def reporthook(blocknum, blocksize, totalsize):
            if show_progress and totalsize > 0:
                percent = min(100, blocknum * blocksize * 100 / totalsize)
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)

        urllib.request.urlretrieve(
            artifact['url'],
            tarball_path,
            reporthook=reporthook if show_progress else None
        )

        if show_progress:
            print("\nDownload complete. Verifying...")

        # Verify SHA256 hash
        actual_hash = compute_sha256(tarball_path)
        expected_hash = artifact['sha256']

        if actual_hash != expected_hash:
            tarball_path.unlink()
            raise RuntimeError(
                f"SHA256 hash mismatch for {artifact_name}!\n"
                f"Expected: {expected_hash}\n"
                f"Got:      {actual_hash}\n"
                "The download may be corrupted. Please try again."
            )

        if show_progress:
            print("Hash verified. Extracting...")

        # Extract tarball
        artifact_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tarball_path, 'r:gz') as tar:
            # Security check: ensure all paths are within artifact_dir
            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                prefix = os.path.commonprefix([abs_directory, abs_target])
                return prefix == abs_directory

            members = tar.getmembers()
            for member in members:
                member_path = artifact_dir / member.name
                if not is_within_directory(artifact_dir, member_path):
                    raise RuntimeError(f"Unsafe path in tarball: {member.name}")

            tar.extractall(artifact_dir)

        # Clean up tarball
        tarball_path.unlink()

        if show_progress:
            print(f"Artifact {artifact_name} installed to {artifact_dir}")

        # Verify extracted files exist
        all_exist = all((extract_dir / f).exists() for f in artifact['files'])
        if not all_exist:
            raise RuntimeError(
                f"Extraction incomplete for {artifact_name}. "
                f"Missing files in {extract_dir}"
            )

        return artifact_dir

    except Exception as e:
        # Clean up on failure
        if tarball_path.exists():
            tarball_path.unlink()
        raise RuntimeError(f"Failed to download {artifact_name}: {e}") from e


def get_artifact_path(
    artifact_name: str,
    auto_download: bool = True
) -> Optional[Path]:
    """
    Get path to an artifact, optionally downloading if needed.

    This is the main interface for accessing artifacts in Korg.
    It checks if the artifact exists locally, and if not (and auto_download
    is True), downloads it automatically.

    Parameters
    ----------
    artifact_name : str
        Name of artifact
    auto_download : bool, optional
        Automatically download if not present (default: True)

    Returns
    -------
    Path or None
        Path to artifact directory, or None if not found and auto_download=False

    Examples
    --------
    >>> # Get MARCS atmospheres (downloads if needed)
    >>> path = get_artifact_path('SDSS_MARCS_atmospheres_v2')
    >>> h5_file = path / 'SDSS_MARCS_atmospheres' / 'SDSS_MARCS_atmospheres.h5'
    """
    if artifact_name not in ARTIFACTS:
        raise ValueError(f"Unknown artifact: {artifact_name}")

    artifact = ARTIFACTS[artifact_name]
    data_dir = get_korg_data_dir()
    artifact_dir = data_dir / artifact['git_tree_sha1']
    extract_dir = artifact_dir / artifact['extract_dir']

    # Check if artifact exists
    if extract_dir.exists():
        # Check if it's a placeholder
        main_file = extract_dir / artifact['files'][0]
        if is_placeholder_file(main_file):
            if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
                # In CI with placeholder, return the path but warn
                warnings.warn(
                    f"Using placeholder for {artifact_name} in CI. "
                    "Full functionality not available.",
                    UserWarning
                )
                return artifact_dir

        all_exist = all((extract_dir / f).exists() for f in artifact['files'])
        if all_exist:
            return artifact_dir

    # Artifact doesn't exist
    if not auto_download:
        return None

    # Download it
    return download_artifact(artifact_name, show_progress=True)


def create_placeholder_artifact(artifact_name: str) -> Path:
    """
    Create a placeholder artifact for CI environments.

    This creates empty directory structure with 0-byte files so that
    tests can import modules that reference artifacts, even though
    the actual data isn't present.

    Parameters
    ----------
    artifact_name : str
        Name of artifact

    Returns
    -------
    Path
        Path to placeholder artifact directory

    Examples
    --------
    >>> # In CI setup script:
    >>> create_placeholder_artifact('SDSS_MARCS_atmospheres_v2')
    """
    if artifact_name not in ARTIFACTS:
        raise ValueError(f"Unknown artifact: {artifact_name}")

    artifact = ARTIFACTS[artifact_name]
    data_dir = get_korg_data_dir()
    artifact_dir = data_dir / artifact['git_tree_sha1']
    extract_dir = artifact_dir / artifact['extract_dir']

    # Create directory structure
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Create placeholder files
    for filename in artifact['files']:
        placeholder = extract_dir / filename
        placeholder.touch()

    print(f"Created placeholder for {artifact_name} at {extract_dir}")
    return artifact_dir


def list_artifacts() -> Dict[str, bool]:
    """
    List all artifacts and their availability status.

    Returns
    -------
    dict
        Dictionary mapping artifact names to availability (True/False)

    Examples
    --------
    >>> status = list_artifacts()
    >>> for name, available in status.items():
    ...     print(f"{name}: {'✓' if available else '✗'}")
    """
    data_dir = get_korg_data_dir()
    status = {}

    for name, artifact in ARTIFACTS.items():
        artifact_dir = data_dir / artifact['git_tree_sha1']
        extract_dir = artifact_dir / artifact['extract_dir']

        if not extract_dir.exists():
            status[name] = False
        else:
            # Check if all expected files exist
            all_exist = all((extract_dir / f).exists() for f in artifact['files'])
            # Check if it's a placeholder
            main_file = extract_dir / artifact['files'][0]
            is_placeholder = is_placeholder_file(main_file) if main_file.exists() else True

            status[name] = all_exist and not is_placeholder

    return status

"""Get esbmtk version information.

This is in a dedicated file to avoid circular imports
during initialization
"""
from importlib.metadata import version, PackageNotFoundError


def get_version():
    """Retrieve Version Data"""
    try:
        __version__ = version("esbmtk")
    except PackageNotFoundError:
        __version__ = "unknown"

    return __version__

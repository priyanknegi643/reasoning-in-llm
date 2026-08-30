import importlib


def _is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def is_liger_kernel_available() -> bool:
    """Check whether ``liger_kernel`` is available."""
    return _is_package_available("liger_kernel")

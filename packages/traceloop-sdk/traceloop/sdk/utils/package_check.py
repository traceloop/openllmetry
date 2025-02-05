from importlib.metadata import distributions


def _get_package_name(dist):
    # Try both 'Name' and 'name' keys to handle different metadata formats
    if hasattr(dist, 'metadata') and dist.metadata is not None:
        for key in ('Name', 'name'):
            try:
                return dist.metadata[key].lower()
            except (KeyError, AttributeError):
                continue

    # If metadata is missing or neither key exists, use the distribution name directly
    return dist.name.lower()


installed_packages = {_get_package_name(dist) for dist in distributions()}


def is_package_installed(package_name: str) -> bool:
    return package_name.lower() in installed_packages

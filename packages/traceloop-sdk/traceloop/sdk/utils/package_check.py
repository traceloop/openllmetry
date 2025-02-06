from importlib.metadata import Distribution, distributions


def _get_package_name(dist: Distribution) -> str | None:
    # Try both 'Name' and 'name' keys to handle different metadata formats
    if hasattr(dist, 'metadata') and dist.metadata is not None:
        for key in ('Name', 'name'):
            try:
                return dist.metadata[key].lower()
            except (KeyError, AttributeError):
                continue


installed_packages = {name for dist in distributions() if (name := _get_package_name(dist)) is not None}


def is_package_installed(package_name: str) -> bool:
    return package_name.lower() in installed_packages

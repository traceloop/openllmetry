from importlib.metadata import distributions

installed_packages = {dist.metadata["Name"].lower() for dist in distributions()}

print(installed_packages)


def is_package_installed(package_name: str) -> bool:
    return package_name.lower() in installed_packages

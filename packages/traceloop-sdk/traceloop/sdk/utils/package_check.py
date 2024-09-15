import pkg_resources

installed_packages = {p.key for p in pkg_resources.working_set}


def is_package_installed(package_name: str) -> bool:
    # return importlib.util.find_spec(package_name) is not None
    return package_name in installed_packages

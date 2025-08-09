
def cameltosnake(camel_string: str) -> str:
    if not camel_string:
        return ""
    elif camel_string[0].isupper():
        return f"_{camel_string[0].lower()}{cameltosnake(camel_string[1:])}"
    else:
        return f"{camel_string[0]}{cameltosnake(camel_string[1:])}"

def camel_to_snake(s):
    if len(s) <= 1:
        return s.lower()
    return cameltosnake(s[0].lower() + s[1:])

def is_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return True
    except Exception:
        return False

# --- Optional dependency utility ---
from .package_check import is_package_installed

def require_dependency(package: str, extra: str = None):
    """
    Raise an ImportError with a clear message if a required optional dependency is missing.

    Usage:
        from traceloop.sdk.utils import require_dependency
        require_dependency("openai", "llm")

    This will raise:
        ImportError: Missing optional dependency 'openai'. Please install with 'traceloop-sdk[llm]' to use this feature.
    """
    if not is_package_installed(package):
        if extra:
            raise ImportError(
                f"Missing optional dependency '{package}'. Please install with 'traceloop-sdk[{extra}]' to use this feature."
            )
        else:
            raise ImportError(
                f"Missing optional dependency '{package}'. Please install the appropriate extra to use this feature."
            )

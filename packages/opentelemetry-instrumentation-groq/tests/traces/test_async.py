# test_direct_instrumentation.py
import asyncio
import importlib
import inspect

import pytest

from opentelemetry.instrumentation.groq import GroqInstrumentor, WRAPPED_METHODS, WRAPPED_AMETHODS
from groq.resources.chat.completions import Completions, AsyncCompletions

def get_original_method(package_path, method_name):
    module_parts = package_path.split(".")
    module_name = ".".join(module_parts[:-1])
    object_name = module_parts[-1]
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    return getattr(obj, method_name)

def is_wrapped(method, original_method):
    return method != original_method and inspect.isfunction(method)  # Basic check

def is_async_wrapped(method, original_method):
    return method != original_method and inspect.iscoroutinefunction(method)  # Basic check
@pytest.mark.no_auto_instrument 
def test_direct_instrumentation_sync():
    instrumentor = GroqInstrumentor()
    instrumentor.instrument()

    for wrapped_method_info in WRAPPED_METHODS:
        package_path = wrapped_method_info["package"]
        method_name = wrapped_method_info["method"]
        original_method = get_original_method(package_path, method_name)
        current_method = getattr(globals()[package_path.split('.')[-1]], method_name) # Access from current scope
        assert is_wrapped(current_method, original_method), f"Method {method_name} in {package_path} not wrapped."

    instrumentor.uninstrument()

    for wrapped_method_info in WRAPPED_METHODS:
        package_path = wrapped_method_info["package"]
        method_name = wrapped_method_info["method"]
        original_method = get_original_method(package_path, method_name)
        current_method = getattr(globals()[package_path.split('.')[-1]], method_name) # Access from current scope
        assert current_method is original_method, f"Method {method_name} in {package_path} not unwrapped."


@pytest.mark.asyncio
@pytest.mark.no_auto_instrument 
async def test_direct_instrumentation_async():
    instrumentor = GroqInstrumentor()
    instrumentor.instrument()

    for wrapped_method_info in WRAPPED_AMETHODS:
        package_path = wrapped_method_info["package"]
        method_name = wrapped_method_info["method"]
        original_method = get_original_method(package_path, method_name)
        current_method = getattr(globals()[package_path.split('.')[-1]], method_name) # Access from current scope
        assert is_async_wrapped(current_method, original_method), f"Async method {method_name} in {package_path} not wrapped."

    instrumentor.uninstrument()

    for wrapped_method_info in WRAPPED_AMETHODS:
        package_path = wrapped_method_info["package"]
        method_name = wrapped_method_info["method"]
        original_method = get_original_method(package_path, method_name)
        current_method = getattr(globals()[package_path.split('.')[-1]], method_name) # Access from current scope
        assert current_method is original_method, f"Async method {method_name} in {package_path} not unwrapped."
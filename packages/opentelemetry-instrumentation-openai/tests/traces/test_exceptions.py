import functools

import pytest


def test_inner_exception_isnt_caught(openai_client):
    should_wrap = True

    def throw_exception(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if should_wrap:
                raise Exception("Test exception")
            else:
                return f(*args, **kwargs)
            raise Exception("Test exception")

        return wrapper

    with pytest.raises(Exception):
        throw_exception(openai_client.chat.completions.create)(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )

    should_wrap = False


@pytest.mark.vcr
def test_exception_in_instrumentation_suppressed(openai_client):
    should_wrap = True

    def scramble_response(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if should_wrap:
                response = f(*args, **kwargs)
                response = {}
                return response
            else:
                return f(*args, **kwargs)

        return wrapper

    scramble_response(openai_client.chat.completions.create)(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

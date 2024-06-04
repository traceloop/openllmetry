import pytest

from wrapt import wrap_function_wrapper


def test_inner_exception_isnt_caught(openai_client):
    should_wrap = True

    def just_throw(wrapped, instance, args, kwargs):
        if should_wrap:
            raise Exception("Test exception")
        else:
            return wrapped(*args, **kwargs)

    wrap_function_wrapper(
        "openai.resources.chat.completions",
        "Completions.create",
        just_throw,
    )

    with pytest.raises(Exception):
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )

    should_wrap = False


@pytest.mark.vcr
def test_exception_in_instrumentation_suppressed(openai_client):
    should_wrap = True

    def scramble_response(wrapped, instance, args, kwargs):
        if should_wrap:
            response = wrapped(*args, **kwargs)
            response = {}
            return response
        else:
            return wrapped(*args, **kwargs)

    wrap_function_wrapper(
        "openai.resources.chat.completions",
        "Completions.create",
        scramble_response,
    )

    openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

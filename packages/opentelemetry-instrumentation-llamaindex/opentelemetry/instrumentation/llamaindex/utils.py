from functools import wraps

def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            if to_wrap.get("wrapped"):
                print("Already wrapped")
                return wrapped(*args, **kwargs)

            print("Wrapping")
            to_wrap["wrapped"] = True
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer

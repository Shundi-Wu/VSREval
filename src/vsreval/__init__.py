"""vsreval: evaluation bench for video super-resolution."""

__version__ = "0.1.0"


def evaluate(*args, **kwargs):
    from .runner import evaluate as _evaluate

    return _evaluate(*args, **kwargs)


__all__ = ["evaluate", "__version__"]

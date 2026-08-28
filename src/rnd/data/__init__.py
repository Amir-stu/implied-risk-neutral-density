"""Chain sources: synthetic markets with a known answer, and live Yahoo data.

The Yahoo loader is imported lazily so that the package works, and the tests
run, without a network connection or the optional ``yfinance`` dependency.
"""

from .synthetic import LognormalMixture, synthetic_chain

__all__ = ["LognormalMixture", "synthetic_chain", "load_yahoo_chain"]


def load_yahoo_chain(*args, **kwargs):
    """Deferred import of :func:`rnd.data.yahoo.load_yahoo_chain`."""
    from .yahoo import load_yahoo_chain as loader

    return loader(*args, **kwargs)

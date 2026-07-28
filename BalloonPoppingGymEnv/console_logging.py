"""Sending this package's diagnostics to a console, for the entry points.

A library does not configure logging for its process, so the package's modules
only ever call ``logger.info``. Something has to turn that into output, and that
something is whoever owns the process: the CLI, the example scripts, the
notebook. Before this existed only the CLI did it, so calling
``evaluate_scenario`` from anywhere else printed nothing at all.
"""

import logging
import sys

PACKAGE_LOGGER_NAME = "BalloonPoppingGymEnv"


def configure_console_logging(level=logging.INFO, stream=None):
    """Print this package's log records as plain lines, and nothing else's.

    Scoped to the package logger rather than the root one. ``basicConfig`` on the
    root logger opens the same stream to every dependency that propagates a
    record, and RocketPy is in the middle of adding module loggers of its own, so
    a competitor's score would arrive in the middle of the engine's chatter.

    The formatter is bare ``%(message)s`` deliberately. ``basicConfig``'s default
    is ``levelname:name:message``, which turned ``Total reward: 7`` into
    ``INFO:__main__:Total reward: 7``. That line is the visible result of a run
    and is worth keeping the way it was.

    Calling this twice replaces the handler rather than adding a second one, so a
    notebook cell rerun does not double every line.
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for existing in list(package_logger.handlers):
        package_logger.removeHandler(existing)

    handler = logging.StreamHandler(sys.stdout if stream is None else stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    # Already handled here; letting it propagate would print twice under an
    # application that has its own root handler.
    package_logger.propagate = False
    return package_logger

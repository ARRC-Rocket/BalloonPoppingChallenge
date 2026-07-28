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
# Marks the handler this module owns, so repeat calls replace that one and leave
# anything the host installed alone.
CONSOLE_HANDLER_NAME = "BalloonPoppingGymEnv.console"


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

    The threshold is set on the handler as well as on the logger, which is not
    redundant. A record is filtered by the level of the logger it was *emitted
    on*; propagation then hands it to every ancestor handler without rechecking
    any ancestor logger's level. So a descendant left at ``DEBUG`` sends debug
    records straight to this handler, and with the handler at ``NOTSET`` they
    reach stdout. Measured: a child at ``DEBUG`` printed its record even with the
    package logger at ``INFO``.

    Only the handler this module installed is replaced, and it is closed on the
    way out. Clearing the logger's handlers outright would also discard a file,
    JSON or audit handler belonging to whatever is embedding the environment.

    Two things this does take over, which is worth saying plainly because
    preserving handlers is not the same as preserving policy. Propagation is
    turned off, so handlers on the root logger stop receiving this package's
    records; that is what keeps a score from being printed twice, and it is a
    reasonable trade for something an entry point calls. And the logger's own
    threshold is lowered to ``level`` if it had none, or left where it is if the
    host had already set one lower, so a host's DEBUG file handler keeps working
    while the console still shows only ``level`` and above.
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    # Normalise and validate before touching anything. logging accepts "INFO"
    # as well as an int, and the comparison below is arithmetic, so a string
    # level used to raise TypeError *after* the new handler was attached and the
    # old one closed: the caller got an exception and a half-configured logger.
    # A probe handler does the conversion, because setLevel is where logging
    # itself decides what a level name means.
    probe = logging.Handler()
    probe.setLevel(level)
    numeric_level = probe.level

    # getEffectiveLevel, not .level. NOTSET on a non-root logger does not mean
    # "no threshold", it means "ask my ancestors", so a host that set DEBUG on
    # the root still has an effective DEBUG here. Reading the raw attribute and
    # finding NOTSET, then setting INFO, raised the threshold anyway.
    effective_level = package_logger.getEffectiveLevel()

    for existing in list(package_logger.handlers):
        if existing.get_name() == CONSOLE_HANDLER_NAME:
            package_logger.removeHandler(existing)
            existing.close()

    handler = logging.StreamHandler(sys.stdout if stream is None else stream)
    handler.set_name(CONSOLE_HANDLER_NAME)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(numeric_level)
    package_logger.addHandler(handler)

    # Never raise the threshold. A host that arranged for DEBUG here, directly
    # or through an ancestor, did so to feed its own handler, and moving this
    # logger to INFO would silence that handler even though it is left
    # attached. The console threshold is the one on the handler above, so
    # lowering this one costs nothing.
    package_logger.setLevel(min(effective_level, numeric_level))

    package_logger.propagate = False
    return package_logger

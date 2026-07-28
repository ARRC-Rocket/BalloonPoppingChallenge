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

    ``level=logging.NOTSET`` is the one value that does not mean what the name
    suggests. On a handler it means "handle everything", but the logger gate
    comes first, and ``NOTSET`` on a non-root logger means "ask my ancestors",
    whose default is ``WARNING``. So passing it leaves ``INFO`` records dropped
    before they reach the console. Pass ``logging.DEBUG`` for everything. The
    behaviour is left as ``logging`` defines it for each object rather than
    special-cased here, because a single value where this function disagrees
    with the standard library is the worse surprise.

    Nothing on the logger is touched until the new handler exists and has
    accepted ``level``, so a level ``logging`` rejects raises with the logger
    exactly as it was.
    """
    package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    # Built and configured before the logger is touched at all. setLevel is
    # where logging itself decides what a level name means, and it is the call
    # that rejects a bad one; doing it here rather than after the swap is what
    # makes a rejection leave nothing half-done. It also normalises "INFO" to an
    # int, which the arithmetic at the bottom needs.
    #
    # Not named yet, deliberately. set_name is not configuration, it writes to
    # logging's process-wide handler-name registry, and close() deletes whatever
    # that name currently points at without checking it is the handler being
    # closed. Naming this one first meant the second call took the name, then
    # closing the old handler deleted the new one's entry: the handler stayed
    # attached and kept printing, so every output test passed, while
    # getHandlerByName returned None and an incremental dictConfig could no
    # longer find it. Measured. A rejected level had the same effect, which made
    # the atomicity this function claims untrue in the one way the logger's own
    # attributes do not show.
    handler = logging.StreamHandler(sys.stdout if stream is None else stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    numeric_level = handler.level

    # getEffectiveLevel, not .level. NOTSET on a non-root logger does not mean
    # "no threshold", it means "ask my ancestors", so a host that set DEBUG on
    # the root still has an effective DEBUG here. Reading the raw attribute and
    # finding NOTSET, then setting INFO, raised the threshold anyway.
    effective_level = package_logger.getEffectiveLevel()

    # Everything from here down mutates the logger, and none of it can fail.
    for existing in list(package_logger.handlers):
        if existing.get_name() == CONSOLE_HANDLER_NAME:
            package_logger.removeHandler(existing)
            existing.close()

    # Named only now, once the old owner of the name has let go of it.
    handler.set_name(CONSOLE_HANDLER_NAME)
    package_logger.addHandler(handler)

    # Never raise the threshold. A host that arranged for DEBUG here, directly
    # or through an ancestor, did so to feed its own handler, and moving this
    # logger to INFO would silence that handler even though it is left
    # attached. The console threshold is the one on the handler above, so
    # lowering this one costs nothing.
    package_logger.setLevel(min(effective_level, numeric_level))

    package_logger.propagate = False
    return package_logger

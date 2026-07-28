import logging

from gymnasium.envs.registration import register

register(
    id="BalloonPoppingGymEnv/BalloonPoppingEnv-v0",
    entry_point="BalloonPoppingGymEnv.envs:BalloonPoppingEnv",
)

# A library emits and does not print. Without a handler, records at WARNING and
# above reach logging's last-resort handler and land on stderr; this package has
# none today, and a NullHandler keeps that true if one is ever added. Entry
# points call console_logging.configure_console_logging to opt in.
logging.getLogger(__name__).addHandler(logging.NullHandler())

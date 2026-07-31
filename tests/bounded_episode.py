"""Run an episode to its end, or say which step it should have ended on.

A driver written as ``while not (terminated or truncated)`` ends only when the
environment says so. When a regression stops it saying so, the test does not
fail: it keeps calling ``step`` until the CI job's own timeout, which reports
nothing about where the problem is and costs the whole job's budget.

What this covers and what it does not, measured rather than assumed. Against an
environment that never sets either flag it raises immediately. Against one whose
third step blocks for three seconds it waits the three seconds out, because the
bound counts steps and a step that does not return is not a step it has counted.
That second kind is real here: a non-finite command used to leave the solver
inside scipy's RK45 error control, where ``error_norm`` is NaN so
``error_norm < 1`` is false and no step is ever accepted. It is closed at its
cause by the action validation in #115, not by anything in this file.

The horizon is what the environment itself uses to decide a timeout, plus a
small margin so a legitimate ending on the last step is not cut short.
"""

# Steps past the horizon before the run is called stuck. Enough that an ending
# on the final step is reached normally, few enough to fail quickly.
MARGIN_STEPS = 5


def run_episode(env, next_action, observation=None, max_steps=None):
    """Step until the episode ends, and return how it ended.

    ``next_action`` receives the latest observation and returns an action, so a
    fixed action and an agent both fit without the caller building a closure.
    ``observation`` is the one ``reset`` returned, for an agent that reads it.
    """
    limit = max_steps if max_steps is not None else env.num_timesteps + MARGIN_STEPS
    terminated = truncated = False
    info = {}

    for step in range(limit):
        observation, _reward, terminated, truncated, info = env.step(
            next_action(observation)
        )
        if terminated or truncated:
            return step + 1, terminated, truncated, info

    raise AssertionError(
        f"the episode reported neither terminated nor truncated within {limit} "
        f"steps, and the environment's own horizon is {env.num_timesteps}"
    )

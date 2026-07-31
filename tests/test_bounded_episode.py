"""What the bounded driver does, including where it does not help.

The helper it tests is what the golden masters and the lifecycle tests drive
episodes with, so a hole in it is a hole in all of them at once.
"""

import unittest

from tests.bounded_episode import MARGIN_STEPS, run_episode


class _Env:
    """The three attributes `run_episode` reads."""

    def __init__(self, num_timesteps=10, ends_after=None, flags=(True, False)):
        self.num_timesteps = num_timesteps
        self.ends_after = ends_after
        self.flags = flags
        self.calls = 0
        self.actions = []

    def step(self, action):
        self.calls += 1
        self.actions.append(action)
        if self.ends_after is not None and self.calls >= self.ends_after:
            return "observation", 1.0, self.flags[0], self.flags[1], {"n": self.calls}
        return "observation", 0.0, False, False, {}


class TestItStopsWhenTheEpisodeDoes(unittest.TestCase):
    def test_it_returns_the_step_count_and_the_flags(self):
        env = _Env(ends_after=4, flags=(False, True))

        steps, terminated, truncated, info = run_episode(env, lambda _o: "a")

        self.assertEqual((steps, terminated, truncated), (4, False, True))
        self.assertEqual(info, {"n": 4})

    def test_it_stops_calling_step_once_the_episode_has_ended(self):
        """A driver that steps again after the flags are set is the bug the
        runner check exists for, so this one must not be it."""
        env = _Env(ends_after=3)

        run_episode(env, lambda _o: "a")

        self.assertEqual(env.calls, 3)

    def test_the_first_action_sees_the_observation_reset_returned(self):
        """Passing None instead cost an agent that reads it a TypeError."""
        env = _Env(ends_after=1)

        run_episode(env, lambda observation: observation, "from reset")

        self.assertEqual(env.actions, ["from reset"])


class TestItRefusesAnEpisodeThatWillNotEnd(unittest.TestCase):
    def test_flags_that_never_get_set_raise_rather_than_run_on(self):
        env = _Env(num_timesteps=10)

        with self.assertRaisesRegex(AssertionError, "neither terminated nor truncated"):
            run_episode(env, lambda _o: "a")

        self.assertEqual(env.calls, 10 + MARGIN_STEPS)

    def test_the_message_says_what_the_horizon_was(self):
        env = _Env(num_timesteps=7)

        with self.assertRaises(AssertionError) as raised:
            run_episode(env, lambda _o: "a")

        self.assertIn("7", str(raised.exception))

    def test_the_margin_lets_an_episode_end_on_the_last_step(self):
        """The bound must not be the thing that ends a legitimate run."""
        env = _Env(num_timesteps=10, ends_after=10)

        steps, _terminated, _truncated, _info = run_episode(env, lambda _o: "a")

        self.assertEqual(steps, 10)

    def test_a_caller_can_ask_for_a_shorter_bound(self):
        env = _Env(num_timesteps=1000)

        with self.assertRaises(AssertionError):
            run_episode(env, lambda _o: "a", max_steps=3)

        self.assertEqual(env.calls, 3)


class TestWhatTheBoundDoesNotCover(unittest.TestCase):
    """Named so nobody has to rediscover it from a hung CI job."""

    def test_a_single_step_that_blocks_is_not_bounded_by_a_step_count(self):
        """It counts steps, and a step that does not return is not one it has
        counted. That failure is closed at its cause by the action validation in
        #115, not here."""
        import time

        class _Slow(_Env):
            def step(self, action):
                if self.calls == 1:
                    time.sleep(0.2)
                return super().step(action)

        env = _Slow(num_timesteps=3)
        started = time.monotonic()
        with self.assertRaises(AssertionError):
            run_episode(env, lambda _o: "a")

        self.assertGreater(time.monotonic() - started, 0.2)


if __name__ == "__main__":
    unittest.main()

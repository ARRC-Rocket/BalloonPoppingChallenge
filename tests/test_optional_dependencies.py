"""vpython stays an optional extra, in the packaging metadata as well as in code.

`test_vpython_lazy_import.py` pins where the import may appear. That is only half
of what #15 fixed: moving vpython back into the base dependencies would reinstate
the same install for every consumer with the source untouched, and no test would
have noticed.

Parsed with `packaging` rather than by splitting on punctuation. A hand written
reader gets the name out of `vpython>=7.6.5; python_version < '0'` and reports
that the extra provides vpython, when that marker is false on every interpreter
there is and an installer skips the line.
"""

import sys
import unittest
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the floor the package claims
    tomllib = None

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

OPTIONAL_ONLY = "vpython"
EXTRA = "vpython"

# The ends of what the package claims to support, so the extra has to work on
# both rather than on whichever one happens to be running the tests.
SUPPORTED_PYTHONS = ("3.10", "3.14")


def _installs_on(requirement, python_version):
    """Whether an installer would take this line on that interpreter."""
    if requirement.marker is None:
        return True
    environment = dict(default_environment())
    environment["python_version"] = python_version
    environment["python_full_version"] = f"{python_version}.0"
    return requirement.marker.evaluate(environment)


@unittest.skipIf(tomllib is None, "tomllib needs Python 3.11")
class TestVpythonIsAnExtra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]

    def _requirements(self, entries):
        return [Requirement(entry) for entry in entries]

    def test_it_is_not_a_base_dependency(self):
        names = {
            requirement.name.lower()
            for requirement in self._requirements(self.project["dependencies"])
        }

        self.assertNotIn(OPTIONAL_ONLY, names)

    def test_the_extra_exists(self):
        self.assertIn(EXTRA, self.project["optional-dependencies"])

    def test_the_extra_installs_it_on_every_supported_python(self):
        """Named, and actually installed. A marker no interpreter satisfies
        leaves the name in the file and vpython on nobody's machine."""
        entries = self.project["optional-dependencies"][EXTRA]
        wanted = [
            requirement
            for requirement in self._requirements(entries)
            if requirement.name.lower() == OPTIONAL_ONLY
        ]
        self.assertTrue(wanted, f"the {EXTRA} extra does not mention {OPTIONAL_ONLY}")

        for python_version in SUPPORTED_PYTHONS:
            with self.subTest(python=python_version):
                self.assertTrue(
                    any(
                        _installs_on(requirement, python_version)
                        for requirement in wanted
                    ),
                    f"no {OPTIONAL_ONLY} line applies on Python {python_version}",
                )

    def test_a_marker_no_interpreter_satisfies_is_not_provision(self):
        """The control. A reader that splits on punctuation passes this one,
        which is what this file used to do."""
        never = Requirement("vpython>=7.6.5; python_version < '0'")

        self.assertEqual(never.name.lower(), OPTIONAL_ONLY)
        for python_version in SUPPORTED_PYTHONS:
            with self.subTest(python=python_version):
                self.assertFalse(_installs_on(never, python_version))

    def test_a_marker_that_covers_the_range_is(self):
        """The other half of the control: the check must accept a real one."""
        real = Requirement("vpython>=7.6.5")
        windows_only = Requirement("vpython>=7.6.5; sys_platform == 'win32'")

        for python_version in SUPPORTED_PYTHONS:
            with self.subTest(python=python_version):
                self.assertTrue(_installs_on(real, python_version))
                self.assertEqual(
                    _installs_on(windows_only, python_version),
                    sys.platform == "win32",
                )


if __name__ == "__main__":
    unittest.main()

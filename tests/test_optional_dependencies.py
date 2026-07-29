"""vpython stays an optional extra, in the packaging metadata as well as in code.

`test_vpython_lazy_import.py` pins where the import may appear. That is only half
of what #15 fixed: moving vpython back into the base dependencies would reinstate
the same install for every consumer with the source untouched, and no test would
have noticed.
"""

import sys
import unittest
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the floor the package claims
    tomllib = None

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

OPTIONAL_ONLY = "vpython"
EXTRA = "vpython"


def _requirement_name(specifier):
    """The distribution name out of a requirement string."""
    for separator in ("[", "<", ">", "=", "!", "~", ";", " "):
        specifier = specifier.split(separator, 1)[0]
    return specifier.strip().lower()


@unittest.skipIf(tomllib is None, "tomllib needs Python 3.11")
class TestVpythonIsAnExtra(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]

    def test_it_is_not_a_base_dependency(self):
        names = {_requirement_name(item) for item in self.project["dependencies"]}

        self.assertNotIn(OPTIONAL_ONLY, names)

    def test_the_extra_exists_and_provides_it(self):
        extras = self.project["optional-dependencies"]
        self.assertIn(EXTRA, extras)
        names = {_requirement_name(item) for item in extras[EXTRA]}

        self.assertIn(OPTIONAL_ONLY, names)

    def test_the_reader_finds_a_name_it_should(self):
        """The control: the two assertions above pass on an empty set too."""
        for specifier, expected in (
            ("vpython>=7.6.5", "vpython"),
            ("VPython >= 7", "vpython"),
            ("numpy[extra]==2.0; python_version<'3.12'", "numpy"),
        ):
            with self.subTest(specifier=specifier):
                self.assertEqual(_requirement_name(specifier), expected)


if __name__ == "__main__":
    unittest.main()

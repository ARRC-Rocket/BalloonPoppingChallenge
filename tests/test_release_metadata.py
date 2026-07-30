"""The version and the changelog have to agree.

v0.1.0 and the release after it both said `0.1.0` while the submission format
changed underneath, so the package version could not tell a competitor whether
their file would be accepted. Nothing noticed, because nothing looked.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _declared_version():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml declares no version"
    return match.group(1)


class TestTheVersionIsDocumented(unittest.TestCase):
    def test_the_declared_version_has_a_changelog_section(self):
        version = _declared_version()
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

        self.assertIn(f"## [{version}]", changelog)

    def test_that_section_is_not_empty(self):
        """A heading on its own would pass the test above and say nothing."""
        version = _declared_version()
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        after = changelog.split(f"## [{version}]", 1)[1]
        body = after.split("\n## ", 1)[0]

        self.assertGreater(len(body.strip()), 200, "the section says almost nothing")

    def test_the_section_has_a_comparison_link(self):
        version = _declared_version()
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

        self.assertIn(f"[{version}]: https://", changelog)


if __name__ == "__main__":
    unittest.main()

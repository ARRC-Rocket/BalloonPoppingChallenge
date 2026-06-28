"""Unit tests for the evaluate.py integrity-check helper (issue #35).

`pack_for_submission` warns when the local `evaluate.py` differs from the copy on
main. It hashes the text after dropping a BOM and normalizing line endings, so a
Windows checkout (CRLF, or a UTF-8 BOM) does not flag an otherwise-unmodified
file, while a genuine edit is still caught. These tests pin both halves.

`utils` imports the rocketpy stack at module top, so the import is guarded and
the tests skip cleanly when the simulation stack is absent.
"""

import unittest

try:
    from BalloonPoppingGymEnv.evaluation.results.utils import _normalized_md5

    _STACK_AVAILABLE = True
except ImportError:
    _STACK_AVAILABLE = False

# Reference content. Every "cosmetic" variant must hash the same as this, and
# every "real edit" variant must hash differently.
_BASE = b"def evaluate():\n    return 42\n"

# Variants that differ from _BASE only in BOM / line endings / trailing newline,
# i.e. what a Windows checkout introduces. These must NOT change the hash.
_COSMETIC_VARIANTS = [
    ("lf_baseline", b"def evaluate():\n    return 42\n"),
    ("crlf", b"def evaluate():\r\n    return 42\r\n"),
    ("bom", b"\xef\xbb\xbfdef evaluate():\n    return 42\n"),
    ("bom_crlf", b"\xef\xbb\xbfdef evaluate():\r\n    return 42\r\n"),
    ("no_trailing_newline", b"def evaluate():\n    return 42"),
    ("classic_mac_cr", b"def evaluate():\r    return 42\r"),
]

# Real source edits. These must change the hash, so the check stays honest.
_REAL_EDITS = [
    ("changed_return_value", b"def evaluate():\n    return 43\n"),
    ("extra_inner_space", b"def evaluate():\n    return  42\n"),
    ("added_line", b"def evaluate():\n    x = 1\n    return 42\n"),
    ("renamed_function", b"def evaluate2():\n    return 42\n"),
]


@unittest.skipUnless(_STACK_AVAILABLE, "simulation stack not installed")
class TestNormalizedMd5(unittest.TestCase):
    """`_normalized_md5` ignores BOM/EOL noise but catches real changes (#35)."""

    def test_cosmetic_variants_do_not_change_the_hash(self):
        expected = _normalized_md5(_BASE)
        for label, raw in _COSMETIC_VARIANTS:
            with self.subTest(variant=label):
                self.assertEqual(
                    _normalized_md5(raw),
                    expected,
                    f"{label} should hash like the baseline but did not",
                )

    def test_real_edits_change_the_hash(self):
        expected = _normalized_md5(_BASE)
        for label, raw in _REAL_EDITS:
            with self.subTest(variant=label):
                self.assertNotEqual(
                    _normalized_md5(raw),
                    expected,
                    f"{label} is a real change and must be detected",
                )

    def test_digest_is_a_32_char_hex_string(self):
        self.assertRegex(_normalized_md5(_BASE), r"\A[0-9a-f]{32}\Z")


if __name__ == "__main__":
    unittest.main()

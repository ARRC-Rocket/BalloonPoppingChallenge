"""Unit tests for the evaluate.py integrity-check helper (issue #35).

`pack_for_submission` warns when the local `evaluate.py` differs from the copy on
main. It hashes the text after dropping a BOM and normalizing line endings, so a
Windows checkout (CRLF, or a UTF-8 BOM) does not flag an otherwise-unmodified
file, while a genuine edit is still caught. These tests pin both halves.

`_normalized_digest` is a pure byte helper, but it lives in `results.utils`, which
imports the rocketpy stack at module top, so importing it needs the stack. Gate on
whether rocketpy is installed rather than a blanket `except ImportError`: if the
stack is present but the helper was renamed or `utils` fails to import, that is a
real regression and should fail loudly instead of skipping silently.
"""

import importlib.util
import unittest

_STACK_AVAILABLE = importlib.util.find_spec("rocketpy") is not None

if _STACK_AVAILABLE:
    from BalloonPoppingGymEnv.evaluation.results.utils import _normalized_digest

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
class TestNormalizedDigest(unittest.TestCase):
    """`_normalized_digest` ignores BOM/EOL noise but catches real changes (#35)."""

    def test_cosmetic_variants_do_not_change_the_hash(self):
        expected = _normalized_digest(_BASE)
        for label, raw in _COSMETIC_VARIANTS:
            with self.subTest(variant=label):
                self.assertEqual(
                    _normalized_digest(raw),
                    expected,
                    f"{label} should hash like the baseline but did not",
                )

    def test_real_edits_change_the_hash(self):
        expected = _normalized_digest(_BASE)
        for label, raw in _REAL_EDITS:
            with self.subTest(variant=label):
                self.assertNotEqual(
                    _normalized_digest(raw),
                    expected,
                    f"{label} is a real change and must be detected",
                )

    def test_digest_is_a_64_char_hex_string(self):
        self.assertRegex(_normalized_digest(_BASE), r"\A[0-9a-f]{64}\Z")

    def test_form_feed_is_not_treated_as_a_line_ending(self):
        # ``str.splitlines()`` breaks on a superset of the tokenizer's physical
        # line endings, so hashing via splitlines let an LF -> form-feed swap keep
        # the same hash while changing the program. Here the baseline sets
        # ``result``; the form-feed variant folds the assignment into the comment,
        # so the two must not hash alike. Form feed stands in for the whole
        # splitlines-only class (vertical tab, NEL, U+2028, ...).
        baseline = b"# set result\nresult = 1\n"
        semantic_edit = b"# set result\fresult = 1\n"
        self.assertNotEqual(
            _normalized_digest(semantic_edit),
            _normalized_digest(baseline),
            "form feed must not be normalized away like a real line ending",
        )

    def test_non_utf8_valid_python_source_does_not_hash_crash(self):
        # A ``coding:`` declaration makes non-UTF-8 bytes valid Python source. The
        # integrity check must still produce a digest for it (a mismatch warning is
        # the point) rather than raising UnicodeDecodeError on the way.
        source = b"# -*- coding: latin-1 -*-\nname = 'caf\xe9'\n"
        compile(source, "<latin1-source>", "exec")
        self.assertRegex(_normalized_digest(source), r"\A[0-9a-f]{64}\Z")


if __name__ == "__main__":
    unittest.main()

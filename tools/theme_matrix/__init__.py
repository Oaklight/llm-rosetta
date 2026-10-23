"""The admin panel's theme matrix — what the stylesheet actually renders.

The contract test (``tests/js/theme-contract.test.mjs``) reads the CSS as text
and says what a scheme is *allowed* to say.  This harness looks at the result:
every scheme, in both modes, at three widths, against a live panel.  It lives
under ``tools/`` because it needs a browser and a running gateway, so it is
never part of the library's test suite.
"""

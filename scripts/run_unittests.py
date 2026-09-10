"""Run all tests or selected unittest names without importing Blender's entry point."""

from pathlib import Path
import sys
import types
import unittest


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "tests"), str(root)]
    package = types.ModuleType("match_perspective")
    package.__path__ = [str(root)]
    package.__file__ = str(root / "__init__.py")
    sys.modules["match_perspective"] = package
    loader = unittest.TestLoader()
    suite = (
        loader.loadTestsFromNames(sys.argv[1:])
        if len(sys.argv) > 1
        else loader.discover(str(root / "tests"))
    )
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

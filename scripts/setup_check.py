#!/usr/bin/env python
import argparse
import importlib
import os
import sys


def check_python():
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        return False, f"Python >=3.9 required, found {major}.{minor}"
    return True, f"Python version {major}.{minor}"


def check_import(package: str):
    try:
        importlib.import_module(package)
        return True, f"Import '{package}' OK"
    except Exception as e:  # pragma: no cover - just diagnostic
        return False, f"Import '{package}' failed: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify project setup")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")
    args = parser.parse_args()

    ok_python, msg_python = check_python()
    ok_import, msg_import = check_import("nova_retrieval_vlm")

    env_ok = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))
    msg_env = "API key found" if env_ok else "No API key set"

    all_ok = ok_python and ok_import

    if args.verbose:
        for ok, msg in [(ok_python, msg_python), (ok_import, msg_import)]:
            prefix = "✅" if ok else "❌"
            print(f"{prefix} {msg}")
        prefix = "✅" if env_ok else "⚠️"
        print(f"{prefix} {msg_env}")

    if not all_ok and args.fix:
        # placeholder for future fixes
        print("Attempting to fix issues (not implemented)")

    print("Setup check", "passed" if all_ok else "failed")
    return 0 if all_ok else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

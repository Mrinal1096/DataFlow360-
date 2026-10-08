"""
DataFlow360 — Environment Verification Script
=============================================
Purpose : Confirms that the Phase 1 environment is correctly configured.
Run with: python verify_setup.py
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Verify Python version meets project requirements."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"  Python version  : {version_str}")

    if version.major == 3 and version.minor >= 11:
        print("  Python check    : PASSED")
        return True
    else:
        print("  Python check    : FAILED — expected Python 3.11 or higher")
        return False


def check_virtual_environment():
    """Confirm script is running inside .venv, not global Python."""
    venv_path = sys.prefix
    print(f"  Python location : {venv_path}")

    if ".venv" in venv_path:
        print("  Venv check      : PASSED — running inside .venv")
        return True
    else:
        print("  Venv check      : WARNING — not running inside .venv")
        return False


def check_dotenv():
    """Verify python-dotenv is installed and can load .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        project_name = os.getenv("PROJECT_NAME", "NOT FOUND")
        environment = os.getenv("ENVIRONMENT", "NOT FOUND")
        import importlib.metadata
        dotenv_version = importlib.metadata.version("python-dotenv")
        print(f"  python-dotenv   : INSTALLED (v{dotenv_version})")
        print(f"  PROJECT_NAME    : {project_name}")
        print(f"  ENVIRONMENT     : {environment}")
        print("  .env check      : PASSED")
        return True
    except ImportError:
        print("  python-dotenv   : FAILED — not installed")
        return False


def check_project_structure():
    """Confirm all required project folders exist."""
    required_folders = [
        "src/ingestion",
        "src/transformation",
        "src/validation",
        "src/pipeline",
        "src/database",
        "src/dashboard",
        "data/raw",
        "data/processed",
        "data/sample",
        "tests",
        "config",
        "logs",
        "notebooks",
        "docs",
    ]

    all_present = True
    for folder in required_folders:
        exists = Path(folder).is_dir()
        status = "OK" if exists else "MISSING"
        print(f"  {folder:<30} : {status}")
        if not exists:
            all_present = False

    if all_present:
        print("  Folder check    : PASSED")
    else:
        print("  Folder check    : FAILED — some folders are missing")

    return all_present


def main():
    print("=" * 55)
    print("  DataFlow360 — Environment Verification")
    print("=" * 55)

    print("\n[1] Python Environment")
    python_ok = check_python_version()
    venv_ok = check_virtual_environment()

    print("\n[2] python-dotenv & .env File")
    dotenv_ok = check_dotenv()

    print("\n[3] Project Folder Structure")
    structure_ok = check_project_structure()

    print("\n" + "=" * 55)
    if all([python_ok, venv_ok, dotenv_ok, structure_ok]):
        print("  RESULT: ALL CHECKS PASSED — Phase 1 Complete!")
    else:
        print("  RESULT: SOME CHECKS FAILED — review output above.")
    print("=" * 55)


if __name__ == "__main__":
    main()

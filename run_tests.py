#!/usr/bin/env python
import os
import sys
import subprocess
from pathlib import Path

# Add the project root and src directories to Python path
project_root = Path(__file__).parent.absolute()
src_dir = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Print the current Python path for debugging
print("Python path:")
for path in sys.path:
    print(f"  - {path}")

# Run a simple test to verify imports work
print("\nVerifying imports...")
try:
    from nova_retrieval_vlm.config import Config
    print("✅ Import successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Run the tests
print("\nRunning tests...")
test_dir = project_root / "tests"
test_files = [f for f in test_dir.glob("test_*.py") if f.is_file()]

if not test_files:
    print("No test files found!")
    sys.exit(1)

for test_file in test_files:
    print(f"\n--- Running tests in {test_file.name} ---")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        env={**os.environ, "PYTHONPATH": f"{src_dir}:{project_root}:{os.environ.get('PYTHONPATH', '')}"}
    )
    if result.returncode != 0:
        print(f"❌ Tests in {test_file.name} failed!")
    else:
        print(f"✅ Tests in {test_file.name} passed!")

print("\nTest run completed.") 
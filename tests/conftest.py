import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Avoid albumentations version check warnings in offline environments.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

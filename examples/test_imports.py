"""
SAPPHIRE Simple Test
"""

# Test basic imports
print("Testing imports...")

try:
    import numpy as np
    print("✅ numpy")
except ImportError as e:
    print(f"❌ numpy: {e}")

try:
    import pandas as pd
    print("✅ pandas")
except ImportError as e:
    print(f"❌ pandas: {e}")

try:
    import scanpy as sc
    print("✅ scanpy")
except ImportError as e:
    print(f"❌ scanpy: {e}")

# Test SAPPHIRE import
try:
    import sys
    sys.path.insert(0, '/Users/ziye/Documents/sapphire_package')
    import sapphire
    print(f"✅ SAPPHIRE version: {sapphire.__version__}")
except ImportError as e:
    print(f"❌ SAPPHIRE: {e}")

print("\n" + "="*50)
print("If all show ✅, ready to test!")
print("="*50)
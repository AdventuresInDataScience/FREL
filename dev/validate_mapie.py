"""
Simple validation test for MAPIE wrapper - just checks imports and structure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("Testing MAPIE wrapper imports and structure...")

# Test imports
try:
    from mapie import (
        SklearnTensorFlowWrapper,
        SklearnLightGBMWrapper,
        MapiePredictor,
        create_mapie_predictor_from_model
    )
    print("✓ All classes imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test class structure
print("\nChecking class methods...")

# SklearnTensorFlowWrapper
assert hasattr(SklearnTensorFlowWrapper, 'fit'), "Missing fit method"
assert hasattr(SklearnTensorFlowWrapper, 'predict'), "Missing predict method"
assert hasattr(SklearnTensorFlowWrapper, '_split_X'), "Missing _split_X method"
print("✓ SklearnTensorFlowWrapper has required methods")

# SklearnLightGBMWrapper
assert hasattr(SklearnLightGBMWrapper, 'fit'), "Missing fit method"
assert hasattr(SklearnLightGBMWrapper, 'predict'), "Missing predict method"
print("✓ SklearnLightGBMWrapper has required methods")

# MapiePredictor
assert hasattr(MapiePredictor, 'fit'), "Missing fit method"
assert hasattr(MapiePredictor, 'predict_intervals'), "Missing predict_intervals method"
assert hasattr(MapiePredictor, 'predict_single_interval'), "Missing predict_single_interval method"
assert hasattr(MapiePredictor, 'predict_point'), "Missing predict_point method"
assert hasattr(MapiePredictor, '_combine_inputs'), "Missing _combine_inputs method"
print("✓ MapiePredictor has required methods")

# Test function exists
assert callable(create_mapie_predictor_from_model), "Missing factory function"
print("✓ create_mapie_predictor_from_model function exists")

print("\n" + "="*60)
print("✓ All validation checks passed!")
print("="*60)
print("\nThe MAPIE wrapper is ready to use.")
print("\nNext steps:")
print("1. See MAPIE_GUIDE.md for usage examples")
print("2. Run: uv run python dev/example_mapie.py (for full examples)")
print("3. Import in your code: from src.mapie import MapiePredictor")

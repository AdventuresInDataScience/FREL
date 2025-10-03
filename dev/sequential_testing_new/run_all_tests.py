"""
Test Suite Runner - Execute All Tests in Sequence
"""

import sys
import subprocess
from pathlib import Path
import time

# Test files in dependency order
TEST_FILES = [
    "01_test_data.py",
    "02_test_forward_windows.py",
    "03_test_scale.py",
    "04_test_synth.py",  # Will be created next
    "05a_test_reward_helpers.py",
    "05b_test_reward_simulation.py",  # Will be created next
    "05c_test_reward_metrics.py",  # Will be created next
    "06_test_dataset.py",  # Will be created next
]

def run_test(test_file: Path) -> tuple[bool, float]:
    """Run a single test file."""
    print(f"\n{'='*80}")
    print(f"Running: {test_file.name}")
    print(f"{'='*80}\n")
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(test_file)],
        capture_output=False
    )
    elapsed = time.time() - start
    
    success = (result.returncode == 0)
    return success, elapsed


def main():
    """Run all tests in sequence."""
    test_dir = Path(__file__).parent
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUITE - DUAL POSITION STRUCTURE")
    print("="*80)
    
    results = []
    total_start = time.time()
    
    for test_name in TEST_FILES:
        test_path = test_dir / test_name
        
        if not test_path.exists():
            print(f"\n⚠️  Test file not found: {test_name}")
            print(f"   Skipping...")
            results.append((test_name, None, 0))
            continue
        
        success, elapsed = run_test(test_path)
        results.append((test_name, success, elapsed))
        
        if not success:
            print(f"\n❌ Test failed: {test_name}")
            print(f"   Stopping test suite.")
            break
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    print(f"\n{'Test File':<40} {'Status':<10} {'Time (s)':<10}")
    print("-" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, success, elapsed in results:
        if success is None:
            status = "⏭️  SKIPPED"
            skipped += 1
        elif success:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"{test_name:<40} {status:<10} {elapsed:<10.2f}")
    
    print("-" * 60)
    print(f"{'Total':<40} {passed}/{len(TEST_FILES)} passed")
    print(f"\nTotal time: {total_elapsed:.2f}s")
    
    if failed > 0:
        print(f"\n❌ {failed} test(s) failed")
        sys.exit(1)
    elif skipped > 0:
        print(f"\n⚠️  {skipped} test(s) skipped")
        sys.exit(2)
    else:
        print(f"\n✅ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

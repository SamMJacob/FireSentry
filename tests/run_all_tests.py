#!/usr/bin/env python3
"""
Master Test Runner

Runs all test suites in the correct sequence.
Provides comprehensive testing before running the full pipeline.

Usage:
    python tests/run_all_tests.py
"""

import sys
from pathlib import Path
import time
import subprocess

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class MasterTestRunner:
    """Master test runner for all test suites."""
    
    def __init__(self):
        self.test_suites = [
            ("Data Integrity", "tests/test_data_integrity.py"),
            ("CHIRPS Quick", "tests/test_chirps_quick.py"),
            ("DTW Algorithm", "tests/test_dtw.py"),
            ("Feature Extraction", "tests/test_features.py"),
            ("Pipeline Integration", "tests/test_pipeline.py"),
            ("Model Training", "tests/test_model.py"),
        ]
        self.results = {}
        
    def run_test_suite(self, name, script_path):
        """Run a single test suite."""
        print(f"\n{'='*80}")
        print(f"RUNNING {name.upper()} TEST SUITE")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Run the test script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test suite
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {name} tests PASSED ({duration:.1f}s)")
                self.results[name] = {"status": "PASSED", "duration": duration}
                return True
            else:
                print(f"❌ {name} tests FAILED ({duration:.1f}s)")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                self.results[name] = {"status": "FAILED", "duration": duration}
                return False
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ {name} tests TIMEOUT ({duration:.1f}s)")
            self.results[name] = {"status": "TIMEOUT", "duration": duration}
            return False
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {name} tests ERROR ({duration:.1f}s): {e}")
            self.results[name] = {"status": "ERROR", "duration": duration}
            return False
    
    def run_all_tests(self):
        """Run all test suites in sequence."""
        print("\n" + "="*80)
        print("FIRESENTRY COMPREHENSIVE TEST SUITE")
        print("="*80)
        print("Running all test suites to verify pipeline readiness...")
        
        start_time = time.time()
        passed = 0
        total = len(self.test_suites)
        
        for name, script_path in self.test_suites:
            if self.run_test_suite(name, script_path):
                passed += 1
        
        total_duration = time.time() - start_time
        
        # Summary
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nTest Suite Results:")
        for name, result in self.results.items():
            status_icon = {
                "PASSED": "✅",
                "FAILED": "❌",
                "TIMEOUT": "⏰",
                "ERROR": "💥"
            }.get(result["status"], "❓")
            
            print(f"  {status_icon} {name}: {result['status']} ({result['duration']:.1f}s)")
        
        print(f"\nOverall Results:")
        print(f"  Passed: {passed}/{total}")
        print(f"  Total Duration: {total_duration:.1f}s")
        
        if passed == total:
            print(f"\n🎉 ALL TESTS PASSED! Pipeline is ready for full execution.")
            print(f"\nNext Steps:")
            print(f"  1. Run full feature pipeline: python scripts/build_features.py")
            print(f"  2. Train model with full dataset")
            print(f"  3. Deploy API endpoint")
            return True
        else:
            print(f"\n⚠️  SOME TESTS FAILED! Fix issues before running full pipeline.")
            print(f"\nFailed Test Suites:")
            for name, result in self.results.items():
                if result["status"] != "PASSED":
                    print(f"  - {name}: {result['status']}")
            return False

def main():
    """Main entry point."""
    runner = MasterTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()



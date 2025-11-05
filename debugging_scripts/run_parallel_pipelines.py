#!/usr/bin/env python3
"""
Run Both Pipelines in Parallel

This script runs both the original and optimized pipelines simultaneously
to compare performance and results.

Outputs:
- Original: data/features.parquet
- Optimized: data/optimized/features.parquet

Author: FireSentry Team
"""

import subprocess
import time
import os
from datetime import datetime
import signal
import sys

def run_original_pipeline():
    """Run the original pipeline."""
    print("🚀 Starting ORIGINAL pipeline...")
    print("   Output: data/features.parquet")
    print("   Status: Running in background")
    
    # Run the original build script
    process = subprocess.Popen(
        ["python", "scripts/build_features.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def run_optimized_pipeline():
    """Run the optimized pipeline."""
    print("🚀 Starting OPTIMIZED pipeline...")
    print("   Output: data/optimized/features.parquet")
    print("   Status: Running in background")
    
    # Run the optimized build script
    process = subprocess.Popen(
        ["python", "scripts/build_features_optimized.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def monitor_processes(original_process, optimized_process):
    """Monitor both processes and report status."""
    print("\n" + "="*80)
    print("PARALLEL PIPELINE MONITORING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nProcesses:")
    print("  📊 Original:   PID {original_process.pid}")
    print("  ⚡ Optimized:  PID {optimized_process.pid}")
    print("\nMonitoring... (Press Ctrl+C to stop monitoring)")
    
    start_time = time.time()
    
    try:
        while True:
            # Check if processes are still running
            original_running = original_process.poll() is None
            optimized_running = optimized_process.poll() is None
            
            elapsed = int(time.time() - start_time)
            hours, minutes, seconds = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
            
            print(f"\r⏱️  Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d} | "
                  f"Original: {'🟢' if original_running else '🔴'} | "
                  f"Optimized: {'🟢' if optimized_running else '🔴'}", end="", flush=True)
            
            if not original_running and not optimized_running:
                print("\n\n✅ Both processes completed!")
                break
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring stopped by user")
        print("Processes are still running in background")
        print("Check terminal windows for progress")

def check_results():
    """Check if results files exist."""
    print("\n" + "="*80)
    print("RESULTS CHECK")
    print("="*80)
    
    original_file = "data/features.parquet"
    optimized_file = "data/optimized/features.parquet"
    
    print(f"📊 Original results:   {original_file}")
    if os.path.exists(original_file):
        size = os.path.getsize(original_file) / (1024*1024)  # MB
        print(f"   ✅ Found ({size:.1f} MB)")
    else:
        print("   ❌ Not found")
    
    print(f"⚡ Optimized results:  {optimized_file}")
    if os.path.exists(optimized_file):
        size = os.path.getsize(optimized_file) / (1024*1024)  # MB
        print(f"   ✅ Found ({size:.1f} MB)")
    else:
        print("   ❌ Not found")

def main():
    """Main execution function."""
    print("🔥 FIRESENTRY PARALLEL PIPELINE RUNNER")
    print("="*80)
    print("This will run both pipelines simultaneously:")
    print("  📊 Original:   Full dataset, 3x3 grid search")
    print("  ⚡ Optimized:  Full dataset, lookup table")
    print("="*80)
    
    # Check if original is already running
    try:
        result = subprocess.run(["pgrep", "-f", "build_features"], capture_output=True, text=True)
        if result.stdout.strip():
            print("⚠️  Original pipeline is already running!")
            print("   This will start a second instance")
    except:
        pass  # pgrep not available on Windows
    
    # Start both processes
    original_process = run_original_pipeline()
    time.sleep(2)  # Small delay to avoid conflicts
    optimized_process = run_optimized_pipeline()
    
    # Monitor processes
    monitor_processes(original_process, optimized_process)
    
    # Check results
    check_results()
    
    print("\n🎯 Next steps:")
    print("  1. Compare file sizes and processing times")
    print("  2. Check data quality (NaN counts, feature distributions)")
    print("  3. Choose the better performing pipeline for final run")

if __name__ == "__main__":
    main()

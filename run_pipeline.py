#!/usr/bin/env python3
"""
Pipeline script to run preprocessing, GSAM, and CoTracker in sequence.
Aborts if any script fails.
"""
import subprocess
import sys
import time


def run_script(script_name):
    """
    Run a Python script and return True if successful, False otherwise.
    """
    print(f"\n{'='*70}")
    print(f"🚀 RUNNING: {script_name}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False  # Show output in real-time
        )
        
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS: {script_name} completed in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"❌ ERROR: {script_name} failed after {elapsed:.2f}s")
        print(f"Return code: {e.returncode}")
        print(f"{'='*70}\n")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"❌ ERROR: Unexpected error running {script_name} after {elapsed:.2f}s")
        print(f"Error: {e}")
        print(f"{'='*70}\n")
        return False


def main():
    """
    Run the full pipeline: preprocess -> gsam -> cotracker
    """
    pipeline_start = time.time()
    
    print("\n" + "="*70)
    print("🎬 STARTING FULL PIPELINE")
    print("="*70)
    print("Pipeline order:")
    print("  1. preprocess.py")
    print("  2. process_gsam.py")
    print("  3. process_cotracker.py")
    print("="*70 + "\n")
    
    scripts = [
        "preprocess.py",
        "process_gsam.py",
        "process_cotracker.py"
    ]
    
    for idx, script in enumerate(scripts, 1):
        print(f"\n📍 Step {idx}/{len(scripts)}")
        
        success = run_script(script)
        
        if not success:
            print("\n" + "="*70)
            print(f"⛔ PIPELINE ABORTED at step {idx}/{len(scripts)}")
            print(f"Failed script: {script}")
            print("="*70 + "\n")
            sys.exit(1)
    
    pipeline_elapsed = time.time() - pipeline_start
    
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Total pipeline time: {pipeline_elapsed:.2f}s ({pipeline_elapsed/60:.2f} minutes)")
    print("All scripts executed without errors.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


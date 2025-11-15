#!/usr/bin/env python3
"""
Pipeline script to run preprocessing, GSAM, and CoTracker in sequence.
Aborts if any script fails.
"""
import subprocess
import sys
import time


def run_script(script_name, args=None):
    """
    Run a Python script with optional arguments and return True if successful, False otherwise.
    """
    if args is None:
        args = []
    
    cmd_display = f"{script_name} {' '.join(args)}" if args else script_name
    
    print(f"\n{'='*70}")
    print(f"🚀 RUNNING: {cmd_display}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name] + args,
            check=True,
            capture_output=False  # Show output in real-time
        )
        
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS: {cmd_display} completed in {elapsed:.2f}s")
        print(f"{'='*70}\n")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"❌ ERROR: {cmd_display} failed after {elapsed:.2f}s")
        print(f"Return code: {e.returncode}")
        print(f"{'='*70}\n")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"❌ ERROR: Unexpected error running {cmd_display} after {elapsed:.2f}s")
        print(f"Error: {e}")
        print(f"{'='*70}\n")
        return False


def main():
    """
    Run the full pipeline: preprocess -> gsam (world) -> gsam (human) -> diffueraser -> cotracker
    """
    pipeline_start = time.time()
    
    print("\n" + "="*70)
    print("🎬 STARTING FULL PIPELINE")
    print("="*70)
    print("Pipeline order:")
    print("  1. preprocess.py - Extract and resize frames")
    print("  2. process_gsam.py (human mode) - Detect humans")
    print("  3. process_diffueraser.py - Erase humans from frames")
    print("  4. process_gsam.py (world mode) - Detect objects on clean frames")
    print("  5. process_cotracker.py - Track points on clean frames")
    print("="*70 + "\n")
    
    scripts = [
        ("preprocess.py", []),
        ("process_gsam.py", ["--mode", "human"]),
        ("process_diffueraser.py", []),
        ("process_gsam.py", ["--mode", "world"]),
        ("process_cotracker.py", [])
    ]
    
    for idx, (script, args) in enumerate(scripts, 1):
        print(f"\n📍 Step {idx}/{len(scripts)}")
        
        success = run_script(script, args)
        
        if not success:
            cmd_display = f"{script} {' '.join(args)}" if args else script
            print("\n" + "="*70)
            print(f"⛔ PIPELINE ABORTED at step {idx}/{len(scripts)}")
            print(f"Failed script: {cmd_display}")
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


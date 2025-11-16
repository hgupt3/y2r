#!/usr/bin/env python3
"""
Pipeline script to run preprocessing, GSAM, and CoTracker in sequence.
Aborts if any script fails.
"""
import subprocess
import sys
import time

# ===== PIPELINE STAGE CONFIGURATION =====
# Set to True to run the stage, False to skip it
PIPELINE_CONFIG = {
    "preprocess": False,          # Extract and resize frames from videos
    "gsam_human": True,          # Detect humans in frames
    "diffueraser": True,         # Erase detected humans from frames
    "gsam_world": True,          # Detect objects (e.g., blocks) in clean frames
    "cotracker": True,           # Track points on clean frames
}
# =========================================


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
    print(f"  1. preprocess.py - Extract and resize frames {'[ENABLED]' if PIPELINE_CONFIG['preprocess'] else '[SKIPPED]'}")
    print(f"  2. process_gsam.py (human mode) - Detect humans {'[ENABLED]' if PIPELINE_CONFIG['gsam_human'] else '[SKIPPED]'}")
    print(f"  3. process_diffueraser.py - Erase humans from frames {'[ENABLED]' if PIPELINE_CONFIG['diffueraser'] else '[SKIPPED]'}")
    print(f"  4. process_gsam.py (world mode) - Detect objects on clean frames {'[ENABLED]' if PIPELINE_CONFIG['gsam_world'] else '[SKIPPED]'}")
    print(f"  5. process_cotracker.py - Track points on clean frames {'[ENABLED]' if PIPELINE_CONFIG['cotracker'] else '[SKIPPED]'}")
    print("="*70 + "\n")
    
    # Define all pipeline stages with their config keys
    all_scripts = [
        ("preprocess", "preprocess.py", []),
        ("gsam_human", "process_gsam.py", ["--mode", "human"]),
        ("diffueraser", "process_diffueraser.py", []),
        ("gsam_world", "process_gsam.py", ["--mode", "world"]),
        ("cotracker", "process_cotracker.py", [])
    ]
    
    # Filter to only enabled scripts
    scripts = [(script, args) for config_key, script, args in all_scripts if PIPELINE_CONFIG[config_key]]
    
    if not scripts:
        print("⚠️  WARNING: All pipeline stages are disabled!")
        print("Enable at least one stage in PIPELINE_CONFIG at the top of this file.")
        return
    
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


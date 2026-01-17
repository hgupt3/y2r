#!/usr/bin/env python3
"""
Compare two normalization_stats.yaml files to verify they match within tolerance.
Used to validate that adaptive worker mode produces the same statistics as sequential mode.

Usage:
    python compare_stats.py file1.yaml file2.yaml [--tolerance 0.001]
"""
import yaml
import argparse
import numpy as np
from pathlib import Path


def load_yaml(file_path):
    """Load YAML file and return as dict"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def parse_value(value):
    """Parse a value that might be a number or a string representation of a number"""
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, str):
        # Handle scientific notation and regular floats
        try:
            return float(value)
        except ValueError:
            return None
    return None


def compare_arrays(arr1, arr2, name, tolerance):
    """Compare two arrays (lists) element-wise"""
    if len(arr1) != len(arr2):
        print(f"  ❌ {name}: Different lengths ({len(arr1)} vs {len(arr2)})")
        return False

    all_match = True
    for i, (v1, v2) in enumerate(zip(arr1, arr2)):
        val1 = parse_value(v1)
        val2 = parse_value(v2)

        if val1 is None or val2 is None:
            continue

        abs_diff = abs(val1 - val2)
        rel_diff = abs_diff / (abs(val1) + 1e-10) * 100  # Relative % difference

        if abs_diff > tolerance and rel_diff > 0.1:  # Absolute > tolerance AND relative > 0.1%
            print(f"  ⚠️  {name}[{i}]: {val1:.6f} vs {val2:.6f} (diff={abs_diff:.6f}, {rel_diff:.3f}%)")
            all_match = False

    if all_match:
        print(f"  ✓ {name}: Match (within {tolerance} tolerance)")

    return all_match


def compare_value(val1, val2, name, tolerance):
    """Compare two scalar values"""
    v1 = parse_value(val1)
    v2 = parse_value(val2)

    if v1 is None or v2 is None:
        print(f"  ⚠️  {name}: Non-numeric values")
        return True

    abs_diff = abs(v1 - v2)
    rel_diff = abs_diff / (abs(v1) + 1e-10) * 100

    if abs_diff > tolerance and rel_diff > 0.1:
        print(f"  ⚠️  {name}: {v1:.6f} vs {v2:.6f} (diff={abs_diff:.6f}, {rel_diff:.3f}%)")
        return False
    else:
        print(f"  ✓ {name}: Match (within {tolerance} tolerance)")
        return True


def compare_stats(stats1, stats2, tolerance=0.001):
    """Compare two statistics dictionaries"""
    all_match = True

    # Displacement stats
    print("\n📊 Displacement Statistics:")
    if 'displacement_mean' in stats1 and 'displacement_mean' in stats2:
        if not compare_arrays(stats1['displacement_mean'], stats2['displacement_mean'],
                             'displacement_mean', tolerance):
            all_match = False

    if 'displacement_std' in stats1 and 'displacement_std' in stats2:
        if not compare_arrays(stats1['displacement_std'], stats2['displacement_std'],
                             'displacement_std', tolerance):
            all_match = False

    if 'num_displacement_samples' in stats1 and 'num_displacement_samples' in stats2:
        if stats1['num_displacement_samples'] != stats2['num_displacement_samples']:
            print(f"  ❌ num_displacement_samples: {stats1['num_displacement_samples']} vs {stats2['num_displacement_samples']}")
            all_match = False
        else:
            print(f"  ✓ num_displacement_samples: {stats1['num_displacement_samples']} (exact match)")

    # Depth stats
    print("\n📊 Depth Statistics:")
    if 'depth_mean' in stats1 and 'depth_mean' in stats2:
        if not compare_value(stats1['depth_mean'], stats2['depth_mean'],
                            'depth_mean', tolerance):
            all_match = False

    if 'depth_std' in stats1 and 'depth_std' in stats2:
        if not compare_value(stats1['depth_std'], stats2['depth_std'],
                            'depth_std', tolerance):
            all_match = False

    if 'num_depth_samples' in stats1 and 'num_depth_samples' in stats2:
        if stats1['num_depth_samples'] != stats2['num_depth_samples']:
            print(f"  ❌ num_depth_samples: {stats1['num_depth_samples']} vs {stats2['num_depth_samples']}")
            all_match = False
        else:
            print(f"  ✓ num_depth_samples: {stats1['num_depth_samples']} (exact match)")

    # Pose stats
    print("\n📊 Pose Statistics:")
    if 'pose_mean' in stats1 and 'pose_mean' in stats2:
        if not compare_arrays(stats1['pose_mean'], stats2['pose_mean'],
                             'pose_mean', tolerance):
            all_match = False

    if 'pose_std' in stats1 and 'pose_std' in stats2:
        if not compare_arrays(stats1['pose_std'], stats2['pose_std'],
                             'pose_std', tolerance):
            all_match = False

    if 'num_pose_samples' in stats1 and 'num_pose_samples' in stats2:
        if stats1['num_pose_samples'] != stats2['num_pose_samples']:
            print(f"  ❌ num_pose_samples: {stats1['num_pose_samples']} vs {stats2['num_pose_samples']}")
            all_match = False
        else:
            print(f"  ✓ num_pose_samples: {stats1['num_pose_samples']} (exact match)")

    # Hand pose stats
    print("\n📊 Hand Pose Statistics:")
    if 'hand_uvd_displacement_mean' in stats1 and 'hand_uvd_displacement_mean' in stats2:
        if not compare_arrays(stats1['hand_uvd_displacement_mean'], stats2['hand_uvd_displacement_mean'],
                             'hand_uvd_displacement_mean', tolerance):
            all_match = False

    if 'hand_uvd_displacement_std' in stats1 and 'hand_uvd_displacement_std' in stats2:
        if not compare_arrays(stats1['hand_uvd_displacement_std'], stats2['hand_uvd_displacement_std'],
                             'hand_uvd_displacement_std', tolerance):
            all_match = False

    if 'hand_rot_displacement_mean' in stats1 and 'hand_rot_displacement_mean' in stats2:
        if not compare_arrays(stats1['hand_rot_displacement_mean'], stats2['hand_rot_displacement_mean'],
                             'hand_rot_displacement_mean', tolerance):
            all_match = False

    if 'hand_rot_displacement_std' in stats1 and 'hand_rot_displacement_std' in stats2:
        if not compare_arrays(stats1['hand_rot_displacement_std'], stats2['hand_rot_displacement_std'],
                             'hand_rot_displacement_std', tolerance):
            all_match = False

    if 'num_hand_pose_samples' in stats1 and 'num_hand_pose_samples' in stats2:
        if stats1['num_hand_pose_samples'] != stats2['num_hand_pose_samples']:
            print(f"  ❌ num_hand_pose_samples: {stats1['num_hand_pose_samples']} vs {stats2['num_hand_pose_samples']}")
            all_match = False
        else:
            print(f"  ✓ num_hand_pose_samples: {stats1['num_hand_pose_samples']} (exact match)")

    return all_match


def main():
    parser = argparse.ArgumentParser(description='Compare two normalization_stats.yaml files')
    parser.add_argument('file1', type=str, help='First stats file (e.g., adaptive mode output)')
    parser.add_argument('file2', type=str, help='Second stats file (e.g., sequential mode output)')
    parser.add_argument('--tolerance', type=float, default=0.001,
                       help='Absolute tolerance for numerical comparison (default: 0.001)')

    args = parser.parse_args()

    file1 = Path(args.file1)
    file2 = Path(args.file2)

    if not file1.exists():
        print(f"❌ Error: {file1} does not exist")
        return 1

    if not file2.exists():
        print(f"❌ Error: {file2} does not exist")
        return 1

    print(f"\n{'='*60}")
    print("NORMALIZATION STATS COMPARISON")
    print(f"{'='*60}")
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print(f"Tolerance: {args.tolerance} (absolute), 0.1% (relative)")

    stats1 = load_yaml(file1)
    stats2 = load_yaml(file2)

    all_match = compare_stats(stats1, stats2, tolerance=args.tolerance)

    print(f"\n{'='*60}")
    if all_match:
        print("✅ VALIDATION PASSED: Statistics match within tolerance!")
        print("   Adaptive worker mode produces identical results to sequential mode.")
        print(f"{'='*60}\n")
        return 0
    else:
        print("⚠️  VALIDATION WARNING: Some statistics differ beyond tolerance")
        print("   This may be acceptable if differences are due to floating-point")
        print("   ordering effects from parallel processing. Review differences above.")
        print(f"{'='*60}\n")
        return 1


if __name__ == '__main__':
    exit(main())

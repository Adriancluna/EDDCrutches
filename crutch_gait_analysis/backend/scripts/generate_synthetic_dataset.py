#!/usr/bin/env python3
"""
Generate synthetic crutch fitting dataset from ANSUR II data.

Usage:
    python scripts/generate_synthetic_dataset.py [--subjects 1000]
"""

import argparse
import os
import json
from dataset.synthetic_dataset_generator import SyntheticDatasetGenerator

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--subjects', type=int, default=1000,
                       help='Number of subjects to generate (default: 1000)')
    parser.add_argument('--output', type=str, default='dataset/synthetic_trials.jsonl',
                       help='Output file path (default: dataset/synthetic_trials.jsonl)')

    args = parser.parse_args()

    # Check ANSUR file exists
    ansur_path = 'dataset/ansur_simplified.csv'
    if not os.path.exists(ansur_path):
        print(f"❌ Error: {ansur_path} not found!")
        print("   Please create ansur_simplified.csv first.")
        print("   See Phase 3 documentation.")
        return 1

    print("="*60)
    print("SYNTHETIC DATASET GENERATION")
    print("="*60)

    # Generate
    generator = SyntheticDatasetGenerator(ansur_path)
    trials = generator.generate_dataset(
        n_subjects=args.subjects,
        output_path=args.output
    )

    # Validate output
    print(f"\n🔍 Validating output...")

    with open(args.output, 'r') as f:
        lines = f.readlines()

    print(f"   Lines in file: {len(lines)}")
    print(f"   Expected: {args.subjects * 3}")

    if len(lines) == args.subjects * 3:
        print("   ✅ Line count matches!")
    else:
        print(f"   ⚠️  Line count mismatch")

    # Parse first trial as sample
    sample = json.loads(lines[0])
    print(f"\n📋 Sample trial:")
    print(f"   User: {sample['user_id']}")
    print(f"   Height: {sample['user_height_cm']:.1f} cm")
    print(f"   Grip: #{sample['grip_setting_idx']+1} ({sample['grip_height_cm']:.1f}cm)")
    print(f"   Overall: #{sample['overall_setting_idx']+1} ({sample['overall_length_cm']:.1f}cm)")
    print(f"   Score: {sample['score_total']:.2f}")
    print(f"   Best: {sample['is_best']}")

    print(f"\n✅ Generation complete! Dataset ready for training.")
    return 0

if __name__ == '__main__':
    exit(main())

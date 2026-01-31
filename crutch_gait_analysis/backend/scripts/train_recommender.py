#!/usr/bin/env python3
"""
Train kNN recommender on synthetic dataset.

Usage:
    python scripts/train_recommender.py
"""

import os
import sys
from recommender.knn_recommender import KNNRecommender


def main():
    print("="*60)
    print("kNN RECOMMENDER TRAINING")
    print("="*60)

    # Check dataset exists
    dataset_path = 'dataset/synthetic_trials.jsonl'
    if not os.path.exists(dataset_path):
        print(f"❌ Error: {dataset_path} not found!")
        print("   Run generate_synthetic_dataset.py first")
        return 1

    # Initialize recommender
    print(f"\n📚 Initializing recommender...")
    recommender = KNNRecommender(n_neighbors=5)

    # Load and train
    print(f"\n📥 Loading training data from {dataset_path}")
    recommender.load_training_data(dataset_path)

    print(f"\n🎓 Training models...")
    metrics = recommender.train(dataset_path, test_size=0.2)

    # Test with example users
    print(f"\n\n🧪 TESTING WITH EXAMPLE USERS:")
    print("="*60)

    test_cases = [
        (160, 55),  # Short, light (female typical)
        (175, 75),  # Average male
        (188, 90),  # Tall, heavy male
        (152, 50),  # Very short
        (195, 95),  # Very tall
    ]

    for height, weight in test_cases:
        rec = recommender.predict(height, weight)
        print(f"\n👤 User: {height}cm, {weight}kg")
        print(f"   → Grip: Hole #{rec['grip_setting_number']} ({rec['grip_height_cm']:.1f}cm)")
        print(f"   → Overall: Setting #{rec['overall_setting_number']} ({rec['overall_length_cm']:.1f}cm)")
        print(f"   → Confidence: Grip {rec['grip_confidence']*100:.0f}%, Overall {rec['overall_confidence']*100:.0f}%")

    # Save model
    print(f"\n\n💾 Saving model...")
    recommender.save('models/knn_recommender.pkl')

    # Summary
    print(f"\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"✅ Model trained successfully")
    print(f"   Training samples: {metrics['n_train']}")
    print(f"   Validation samples: {metrics['n_test']}")
    print(f"   Grip MAE: {metrics['mae_grip_cm']:.2f}cm")
    print(f"   Overall MAE: {metrics['mae_overall_cm']:.2f}cm")
    print(f"   Model saved to: models/knn_recommender.pkl")
    print(f"\n✅ Ready to use in main application!")

    return 0


if __name__ == '__main__':
    sys.exit(main())

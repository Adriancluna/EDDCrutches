#!/usr/bin/env python3
"""
k-Nearest Neighbors Recommender for Crutch Settings.

Predicts optimal initial grip and overall height settings based on
user body measurements (height, weight) using similar users from
the training dataset.
"""

import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import os
from config.crutch_models import DEFAULT_CRUTCH


class KNNRecommender:
    """
    k-Nearest Neighbors recommender for initial crutch settings.

    Predicts optimal grip and overall heights from user measurements.

    The recommender works by:
    1. Training on "best" trials from the synthetic dataset
    2. Finding k most similar users (by height/weight) for a new user
    3. Distance-weighting their optimal settings to make predictions

    Attributes:
        n_neighbors: Number of neighbors to consider (default: 5)
        model_grip: Trained kNN model for grip height
        model_overall: Trained kNN model for overall length
        training_data: Dictionary containing training arrays
        is_trained: Whether model has been trained

    Example:
        >>> recommender = KNNRecommender(n_neighbors=5)
        >>> recommender.train('dataset/synthetic_trials.jsonl')
        >>> rec = recommender.predict(user_height_cm=175, user_weight_kg=70)
        >>> recommender.print_recommendation(rec)
    """

    def __init__(self, n_neighbors=5, grip_bias_cm=0.0, overall_bias_cm=6.5):
        """
        Initialize recommender.

        Args:
            n_neighbors: Number of neighbors to use (default: 5)
            grip_bias_cm: Bias correction for grip height (default: 0.0)
                         Positive = recommend higher grip
            overall_bias_cm: Bias correction for overall length (default: 6.5)
                            Positive = recommend taller crutch
                            The training data uses ~73% of height ratio, but
                            clinical formula uses 77%. This 4% gap = ~6.5cm
                            for average height person.
        """
        self.n_neighbors = n_neighbors
        self.grip_bias_cm = grip_bias_cm
        self.overall_bias_cm = overall_bias_cm
        self.model_grip = None
        self.model_overall = None
        self.training_data = None
        self.is_trained = False

    def load_training_data(self, trials_jsonl_path='dataset/synthetic_trials.jsonl'):
        """
        Load and prepare training data from JSONL file.

        Only uses trials marked as is_best=True (optimal settings).

        Args:
            trials_jsonl_path: Path to trials JSONL file

        Returns:
            dict with training arrays (X, y_grip, y_overall, df)

        Raises:
            FileNotFoundError: If trials file doesn't exist
            ValueError: If no best trials found in dataset
        """
        if not os.path.exists(trials_jsonl_path):
            raise FileNotFoundError(f"Training data not found: {trials_jsonl_path}")

        trials = []
        with open(trials_jsonl_path, 'r') as f:
            for line in f:
                trial = json.loads(line)
                if trial.get('is_best', False):  # Only best trials
                    trials.append(trial)

        if len(trials) == 0:
            raise ValueError("No best trials found in dataset!")

        # Convert to DataFrame
        df = pd.DataFrame(trials)

        # Extract features and targets
        X = df[['user_height_cm', 'user_weight_kg']].values
        y_grip = df['grip_height_cm'].values
        y_overall = df['overall_length_cm'].values

        self.training_data = {
            'X': X,
            'y_grip': y_grip,
            'y_overall': y_overall,
            'df': df
        }

        print(f"✓ Loaded {len(trials)} best trials for training")
        print(f"  Height range: {df['user_height_cm'].min():.1f} - {df['user_height_cm'].max():.1f} cm")
        print(f"  Weight range: {df['user_weight_kg'].min():.1f} - {df['user_weight_kg'].max():.1f} kg")

        return self.training_data

    def train(self, trials_jsonl_path='dataset/synthetic_trials.jsonl', test_size=0.2):
        """
        Train kNN models on best trials.

        Trains two separate kNN regressors:
        - One for grip height prediction
        - One for overall length prediction

        Both use distance-weighted voting (closer neighbors count more).

        Args:
            trials_jsonl_path: Path to trials JSONL file
            test_size: Fraction of data for validation (0.2 = 20%)

        Returns:
            dict with training metrics:
                - mae_grip_cm: Mean absolute error for grip predictions
                - mae_overall_cm: Mean absolute error for overall predictions
                - n_train: Number of training samples
                - n_test: Number of test samples
        """
        # Load data
        if self.training_data is None:
            self.load_training_data(trials_jsonl_path)

        X = self.training_data['X']
        y_grip = self.training_data['y_grip']
        y_overall = self.training_data['y_overall']

        # Split train/test
        X_train, X_test, y_grip_train, y_grip_test = train_test_split(
            X, y_grip, test_size=test_size, random_state=42
        )
        _, _, y_overall_train, y_overall_test = train_test_split(
            X, y_overall, test_size=test_size, random_state=42
        )

        # Train models
        print(f"\n🔧 Training kNN models (k={self.n_neighbors})...")

        self.model_grip = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights='distance'  # Closer neighbors weighted more
        )
        self.model_overall = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights='distance'
        )

        self.model_grip.fit(X_train, y_grip_train)
        self.model_overall.fit(X_train, y_overall_train)

        # Validate
        grip_pred = self.model_grip.predict(X_test)
        overall_pred = self.model_overall.predict(X_test)

        mae_grip = mean_absolute_error(y_grip_test, grip_pred)
        mae_overall = mean_absolute_error(y_overall_test, overall_pred)

        print(f"\n📊 Validation Results:")
        print(f"   Grip MAE: {mae_grip:.2f} cm (~{mae_grip/2.54:.2f} inches)")
        print(f"   Overall MAE: {mae_overall:.2f} cm (~{mae_overall/2.54:.2f} inches)")

        if mae_grip < 3.0 and mae_overall < 3.0:
            print(f"   ✅ Accuracy is good (< 3cm error)")
        else:
            print(f"   ⚠️  Accuracy could be better")

        self.is_trained = True

        return {
            'mae_grip_cm': mae_grip,
            'mae_overall_cm': mae_overall,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }

    def predict(self, user_height_cm, user_weight_kg=None, device_profile=None):
        """
        Predict optimal crutch settings for a new user.

        Uses a HYBRID approach:
        - If user is within training data range: Use kNN predictions
        - If user is outside range: Blend with clinical formulas
        - Clinical formulas used as fallback for extrapolation

        Args:
            user_height_cm: User's height in cm
            user_weight_kg: User's weight in kg (optional, defaults to 70)
            device_profile: CrutchDeviceProfile instance (defaults to DEFAULT_CRUTCH)

        Returns:
            dict with recommended settings
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Call .train() first.")

        if user_weight_kg is None:
            user_weight_kg = 70  # Default

        if device_profile is None:
            device_profile = DEFAULT_CRUTCH

        # Get training data range for hybrid approach
        if self.training_data is not None:
            train_height_min = self.training_data['df']['user_height_cm'].min()
            train_height_max = self.training_data['df']['user_height_cm'].max()
        else:
            # Fallback if training data not loaded (model loaded from file)
            train_height_min = 146.0
            train_height_max = 198.0

        # Calculate clinical formula predictions (ground truth)
        # Grip: wrist height ≈ 48.5% of height
        clinical_grip_cm = user_height_cm * 0.485
        # Overall: 77% of height (validated formula)
        clinical_overall_cm = user_height_cm * 0.77

        # Prepare features for kNN
        features = np.array([[user_height_cm, user_weight_kg]])

        # Predict raw values from kNN
        knn_grip_cm = self.model_grip.predict(features)[0]
        knn_overall_cm = self.model_overall.predict(features)[0]

        # Calculate how far outside training range (0 = inside, 1+ = outside)
        if user_height_cm < train_height_min:
            out_of_range_factor = (train_height_min - user_height_cm) / 10.0  # 10cm = fully clinical
        elif user_height_cm > train_height_max:
            out_of_range_factor = (user_height_cm - train_height_max) / 10.0
        else:
            out_of_range_factor = 0.0

        out_of_range_factor = min(1.0, out_of_range_factor)  # Cap at 1.0

        # HYBRID BLENDING: Use clinical formula more as we go out of range
        # Inside range: 100% kNN, Outside range: blend toward clinical
        knn_weight = 1.0 - out_of_range_factor
        clinical_weight = out_of_range_factor

        pred_grip_cm = (knn_grip_cm * knn_weight) + (clinical_grip_cm * clinical_weight)
        pred_overall_cm = (knn_overall_cm * knn_weight) + (clinical_overall_cm * clinical_weight)

        # Apply bias correction ONLY for people within or near training range
        # Don't apply bias for out-of-range people (clinical formula is already good)
        if out_of_range_factor < 0.5:
            pred_grip_cm += self.grip_bias_cm * (1.0 - out_of_range_factor)
            pred_overall_cm += self.overall_bias_cm * (1.0 - out_of_range_factor)

        # Get nearest neighbors for confidence estimation
        distances_grip, indices_grip = self.model_grip.kneighbors(features)
        avg_dist = np.mean(distances_grip)

        # Confidence decreases with distance AND out-of-range factor
        normalized_dist = avg_dist / 25.0
        base_confidence = max(0.5, min(0.95, 1.0 - (normalized_dist * 0.5)))

        # Reduce confidence for out-of-range predictions
        confidence = base_confidence * (1.0 - (out_of_range_factor * 0.3))
        confidence = max(0.5, confidence)

        confidence_grip = confidence
        confidence_overall = confidence

        # Convert to device settings
        grip_info = device_profile.cm_to_setting(pred_grip_cm, 'grip')
        overall_info = device_profile.cm_to_setting(pred_overall_cm, 'overall')

        # Determine method used
        if out_of_range_factor > 0.5:
            method = 'clinical_formula'
        elif out_of_range_factor > 0:
            method = 'knn_hybrid'
        else:
            method = 'knn'

        recommendation = {
            'user_height_cm': user_height_cm,
            'user_weight_kg': user_weight_kg,

            # Grip recommendations
            'grip_height_cm': grip_info['actual_cm'],
            'grip_setting_idx': grip_info['setting_idx'],
            'grip_setting_number': grip_info['setting_number'],
            'grip_confidence': round(confidence_grip, 2),

            # Overall recommendations
            'overall_length_cm': overall_info['actual_cm'],
            'overall_setting_idx': overall_info['setting_idx'],
            'overall_setting_number': overall_info['setting_number'],
            'overall_confidence': round(confidence_overall, 2),

            # Metadata
            'method': method,
            'n_neighbors': self.n_neighbors,
            'out_of_range': out_of_range_factor > 0,
            'out_of_range_factor': round(out_of_range_factor, 2)
        }

        return recommendation

    def save(self, filepath='models/knn_recommender.pkl'):
        """
        Save trained model to file.

        Args:
            filepath: Path to save pickle file

        Raises:
            RuntimeError: If model not trained
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model!")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model_grip': self.model_grip,
            'model_overall': self.model_overall,
            'n_neighbors': self.n_neighbors,
            'grip_bias_cm': self.grip_bias_cm,
            'overall_bias_cm': self.overall_bias_cm,
            'training_data_summary': {
                'n_samples': len(self.training_data['X']),
                'height_range': (
                    self.training_data['df']['user_height_cm'].min(),
                    self.training_data['df']['user_height_cm'].max()
                )
            }
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Model saved to {filepath}")

    def load(self, filepath='models/knn_recommender.pkl'):
        """
        Load trained model from file.

        Args:
            filepath: Path to pickle file

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model_grip = model_data['model_grip']
        self.model_overall = model_data['model_overall']
        self.n_neighbors = model_data['n_neighbors']
        # Load bias corrections (with defaults for backward compatibility)
        # Default overall bias is 6.5cm to correct for training data using 73% ratio
        # instead of clinical 77% ratio
        self.grip_bias_cm = model_data.get('grip_bias_cm', 0.0)
        self.overall_bias_cm = model_data.get('overall_bias_cm', 6.5)
        self.is_trained = True

        print(f"✅ Model loaded from {filepath}")
        print(f"   Trained on {model_data['training_data_summary']['n_samples']} samples")
        print(f"   Bias correction: grip {self.grip_bias_cm:+.1f}cm, overall {self.overall_bias_cm:+.1f}cm")

    def print_recommendation(self, recommendation):
        """
        Pretty-print recommendation to console.

        Args:
            recommendation: dict returned from predict()
        """
        print(f"\n🎯 RECOMMENDED INITIAL SETTINGS")
        print(f"   User: {recommendation['user_height_cm']:.0f}cm, {recommendation['user_weight_kg']:.0f}kg")
        print(f"\n   Handgrip:")
        print(f"      Hole #{recommendation['grip_setting_number']}")
        print(f"      {recommendation['grip_height_cm']:.1f}cm ({recommendation['grip_height_cm']/2.54:.1f}\")")
        print(f"      Confidence: {recommendation['grip_confidence']*100:.0f}%")
        print(f"\n   Overall Length:")
        print(f"      Setting #{recommendation['overall_setting_number']}")
        print(f"      {recommendation['overall_length_cm']:.1f}cm ({recommendation['overall_length_cm']/2.54:.1f}\")")
        print(f"      Confidence: {recommendation['overall_confidence']*100:.0f}%")
        print(f"\n   Method: k-Nearest Neighbors (k={recommendation['n_neighbors']})")


# Convenience function for quick testing
if __name__ == '__main__':
    print("="*60)
    print("KNN RECOMMENDER - QUICK TEST")
    print("="*60)

    # Train
    recommender = KNNRecommender(n_neighbors=5)
    metrics = recommender.train()

    # Test predictions
    print("\n" + "="*60)
    print("TEST PREDICTIONS")
    print("="*60)

    test_cases = [
        (160, 55, "Short, light"),
        (175, 75, "Average"),
        (190, 90, "Tall, heavy"),
    ]

    for height, weight, desc in test_cases:
        print(f"\n--- {desc} ({height}cm, {weight}kg) ---")
        rec = recommender.predict(height, weight)
        print(f"   Grip: #{rec['grip_setting_number']} ({rec['grip_height_cm']:.1f}cm)")
        print(f"   Overall: #{rec['overall_setting_number']} ({rec['overall_length_cm']:.1f}cm)")

    # Save model
    recommender.save('models/knn_recommender.pkl')

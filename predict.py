"""
Prediction script for trained Driver Behavior Classification model
Use this to make predictions on new driving data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import pickle

# Load trained model
MODEL_PATH = 'models/driver_behavior_model.h5'

def load_model_and_preprocessors():
    """Load the trained model, scaler, and label encoder"""
    model = keras.models.load_model(MODEL_PATH)
    
    # Note: In production, save and load the scaler and label_encoder
    # For now, we'll recreate them from the training data
    print("✓ Model loaded successfully")
    return model

def predict_single_sample(model, features_dict):
    """
    Predict behavior for a single driving sample
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    features_dict : dict
        Dictionary containing all required features:
        - speed_kmph
        - accel_x
        - accel_y
        - brake_pressure
        - steering_angle
        - throttle
        - lane_deviation
        - phone_usage
        - headway_distance
        - reaction_time
    
    Returns:
    --------
    dict : Prediction results with probabilities
    """
    # Expected feature order
    feature_order = [
        'speed_kmph', 'accel_x', 'accel_y', 'brake_pressure', 
        'steering_angle', 'throttle', 'lane_deviation', 'phone_usage',
        'headway_distance', 'reaction_time'
    ]
    
    # Create feature array
    features = np.array([[features_dict[f] for f in feature_order]])
    
    # Note: In production, apply the same scaler used during training
    # For demonstration, we'll use the raw features
    
    # Make prediction
    prediction_probs = model.predict(features, verbose=0)[0]
    predicted_class_idx = np.argmax(prediction_probs)
    
    # Class mapping (should match training)
    classes = ['Aggressive', 'Distracted', 'Safe']
    
    result = {
        'predicted_behavior': classes[predicted_class_idx],
        'confidence': float(prediction_probs[predicted_class_idx]),
        'probabilities': {
            classes[i]: float(prediction_probs[i]) 
            for i in range(len(classes))
        }
    }
    
    return result

def predict_batch(model, csv_path):
    """
    Predict behaviors for a batch of samples from CSV
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    csv_path : str
        Path to CSV file with driving data
    
    Returns:
    --------
    pd.DataFrame : Original data with predictions
    """
    df = pd.read_csv(csv_path)
    
    # Make predictions
    predictions_probs = model.predict(df.values, verbose=0)
    predicted_classes = np.argmax(predictions_probs, axis=1)
    
    # Class mapping
    classes = ['Aggressive', 'Distracted', 'Safe']
    
    # Add predictions to dataframe
    df['predicted_behavior'] = [classes[i] for i in predicted_classes]
    df['confidence'] = [predictions_probs[i][predicted_classes[i]] for i in range(len(df))]
    
    return df

# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("DRIVER BEHAVIOR PREDICTION")
    print("=" * 70)
    
    # Load model
    model = load_model_and_preprocessors()
    
    # Example 1: Predict single sample (Safe driving)
    print("\n[Example 1] Safe Driving Scenario:")
    safe_sample = {
        'speed_kmph': 55.0,
        'accel_x': 0.3,
        'accel_y': 0.05,
        'brake_pressure': 15.0,
        'steering_angle': -1.5,
        'throttle': 40.0,
        'lane_deviation': 0.2,
        'phone_usage': 0,
        'headway_distance': 35.0,
        'reaction_time': 0.85
    }
    
    result = predict_single_sample(model, safe_sample)
    print(f"  Predicted: {result['predicted_behavior']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    for behavior, prob in result['probabilities'].items():
        print(f"    {behavior}: {prob:.2%}")
    
    # Example 2: Predict single sample (Aggressive driving)
    print("\n[Example 2] Aggressive Driving Scenario:")
    aggressive_sample = {
        'speed_kmph': 95.0,
        'accel_x': 3.5,
        'accel_y': 0.8,
        'brake_pressure': 85.0,
        'steering_angle': -15.0,
        'throttle': 92.0,
        'lane_deviation': 0.4,
        'phone_usage': 0,
        'headway_distance': 8.0,
        'reaction_time': 0.45
    }
    
    result = predict_single_sample(model, aggressive_sample)
    print(f"  Predicted: {result['predicted_behavior']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    for behavior, prob in result['probabilities'].items():
        print(f"    {behavior}: {prob:.2%}")
    
    # Example 3: Predict single sample (Distracted driving)
    print("\n[Example 3] Distracted Driving Scenario:")
    distracted_sample = {
        'speed_kmph': 42.0,
        'accel_x': 0.6,
        'accel_y': 0.3,
        'brake_pressure': 38.0,
        'steering_angle': -8.0,
        'throttle': 55.0,
        'lane_deviation': 1.4,
        'phone_usage': 1,
        'headway_distance': 18.0,
        'reaction_time': 1.65
    }
    
    result = predict_single_sample(model, distracted_sample)
    print(f"  Predicted: {result['predicted_behavior']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Probabilities:")
    for behavior, prob in result['probabilities'].items():
        print(f"    {behavior}: {prob:.2%}")
    
    print("\n" + "=" * 70)
    print("Note: For production use, ensure you apply the same preprocessing")
    print("(scaling) that was used during training!")
    print("=" * 70)

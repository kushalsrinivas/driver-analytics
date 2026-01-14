# Driver Behavior Classification with TensorFlow

A deep learning model that classifies driver behavior into three categories: **Safe**, **Aggressive**, and **Distracted** using vehicle telemetry data.

## 📊 Dataset

The model is trained on a synthetic dataset with 30,000 labeled samples containing:

### Features (10 total):
- `speed_kmph` - Vehicle speed in km/h
- `accel_x` - Longitudinal acceleration
- `accel_y` - Lateral acceleration
- `brake_pressure` - Brake pressure applied
- `steering_angle` - Steering wheel angle
- `throttle` - Throttle position
- `lane_deviation` - Deviation from lane center
- `phone_usage` - Phone usage indicator (0/1)
- `headway_distance` - Distance to vehicle ahead
- `reaction_time` - Driver reaction time

### Target Classes:
- **Safe**: Smooth acceleration, stable steering, normal reaction times
- **Aggressive**: High speeds, sharp braking/acceleration, short following distances
- **Distracted**: Increased lane deviation, delayed reactions, phone usage

## 🏗️ Model Architecture

**Deep Neural Network (DNN)** with the following layers:
```
Input Layer (10 features)
    ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(32) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Dense(3) + Softmax (Output)
```

**Why this architecture?**
- Deep layers capture complex behavioral patterns
- BatchNormalization for stable training
- Dropout for regularization (prevents overfitting)
- Suitable for tabular data with clear feature interactions

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
```

### Installation

1. Clone or navigate to the project directory:
```bash
cd /path/to/driver
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model

Run the main training script:
```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Split data into train/validation/test sets (70%/15%/15%)
3. Train the neural network
4. Evaluate performance
5. Generate visualizations
6. Save the trained model

**Training time:** ~5-10 minutes on CPU, ~1-2 minutes on GPU

## 📈 Results

After training, you'll find:

### Saved Files:
- `models/driver_behavior_model.h5` - Trained model
- `results/training_history.png` - Accuracy/loss curves
- `results/confusion_matrices.png` - Performance breakdown
- `results/feature_importance.png` - Most important features
- `results/class_performance.png` - Per-class metrics
- `results/prediction_distribution.png` - Label distributions
- `results/training_history.csv` - Training metrics
- `results/test_predictions.csv` - Test set predictions
- `results/performance_metrics.txt` - Summary report

### Expected Performance:
- **Test Accuracy:** ~95-99%
- **Balanced performance** across all three classes

## 🔮 Making Predictions

Use the prediction script for inference:

```python
from predict import load_model_and_preprocessors, predict_single_sample

# Load model
model = load_model_and_preprocessors()

# Example: Safe driving
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
print(f"Predicted: {result['predicted_behavior']}")
print(f"Confidence: {result['confidence']:.2%}")
```

Or run the demo:
```bash
python predict.py
```

## 📁 Project Structure

```
driver/
├── data/
│   └── Driver_Behavior.csv          # Dataset
├── models/
│   └── driver_behavior_model.h5     # Trained model
├── results/
│   ├── *.png                        # Visualizations
│   ├── training_history.csv        # Training logs
│   └── performance_metrics.txt     # Performance report
├── main.py                          # Training script
├── predict.py                       # Prediction script
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🎯 Use Cases

- **ADAS Development**: Behavior-based driver assistance systems
- **Fleet Management**: Monitor driver safety in real-time
- **Insurance Telematics**: Risk assessment and premium calculation
- **Driver Training**: Identify areas for improvement
- **Research**: Benchmark for behavior recognition algorithms

## 🔧 Customization

### Hyperparameter Tuning

Edit these parameters in `main.py`:

```python
BATCH_SIZE = 64        # Batch size for training
EPOCHS = 100           # Maximum epochs
VALIDATION_SPLIT = 0.2 # Validation split ratio
learning_rate = 0.001  # Adam optimizer learning rate
```

### Model Architecture

Modify the `create_model()` function in `main.py` to:
- Add/remove layers
- Change number of neurons
- Adjust dropout rates
- Try different activation functions

### Feature Engineering

Add new features or remove existing ones by modifying the preprocessing section.

## 📊 Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Correct positive predictions per class
- **Recall**: Coverage of actual positives per class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

## ⚠️ Important Notes

1. **Preprocessing**: Always apply the same scaling (StandardScaler) used during training
2. **Balanced Dataset**: Current dataset is balanced; real-world data may require resampling
3. **Synthetic Data**: Trained on synthetic data; fine-tune on real telemetry for production
4. **Real-time Inference**: For streaming data, consider time-series models (LSTM/GRU)

## 🤝 Contributing

Suggestions for improvement:
- [ ] Add LSTM for time-series sequential behavior
- [ ] Implement ensemble methods (combine with XGBoost)
- [ ] Add data augmentation for robustness
- [ ] Create web API for real-time predictions
- [ ] Add explainability (SHAP values)

## 📝 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Dataset: Synthetically generated driver behavior telemetry
- Framework: TensorFlow/Keras
- Visualization: Matplotlib, Seaborn

---

**Questions or issues?** Open an issue or contact the maintainer.
# driver-analytics

"""
Driver Behavior Classification using TensorFlow
Classifies driving behavior into: Safe, Aggressive, or Distracted
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATA_PATH = 'data/Driver_Behavior.csv'
MODEL_SAVE_PATH = 'models/driver_behavior_model.h5'
RESULTS_DIR = 'results'
BATCH_SIZE = 64
EPOCHS = 100
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.15

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("DRIVER BEHAVIOR CLASSIFICATION MODEL")
print("=" * 70)

# ========================
# 1. LOAD AND EXPLORE DATA
# ========================
print("\n[1] Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nFeatures: {list(df.columns[:-1])}")
print(f"Target: {df.columns[-1]}")

# Display basic statistics
print("\n" + "=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(df.head())
print("\nClass Distribution:")
print(df['behavior_label'].value_counts())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# ========================
# 2. DATA PREPROCESSING
# ========================
print("\n[2] Preprocessing data...")

# Separate features and target
X = df.drop('behavior_label', axis=1)
y = df['behavior_label']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
n_classes = len(label_encoder.classes_)

print(f"✓ Classes encoded: {list(label_encoder.classes_)}")
print(f"✓ Number of classes: {n_classes}")

# Split data: Train (70%), Validation (15%), Test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_temp
)

print(f"✓ Train set: {X_train.shape[0]} samples")
print(f"✓ Validation set: {X_val.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled (StandardScaler)")

# ========================
# 3. BUILD MODEL
# ========================
print("\n[3] Building neural network model...")

def create_model(input_dim, n_classes):
    """
    Deep Neural Network for Driver Behavior Classification
    
    Architecture:
    - Input Layer: 10 features
    - Dense Layer 1: 128 neurons + BatchNorm + ReLU + Dropout(0.3)
    - Dense Layer 2: 64 neurons + BatchNorm + ReLU + Dropout(0.3)
    - Dense Layer 3: 32 neurons + BatchNorm + ReLU + Dropout(0.2)
    - Output Layer: 3 neurons + Softmax
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.3, name='dropout_1'),
        
        # Second hidden layer
        layers.Dense(64, activation='relu', name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.3, name='dropout_2'),
        
        # Third hidden layer
        layers.Dense(32, activation='relu', name='dense_3'),
        layers.BatchNormalization(name='bn_3'),
        layers.Dropout(0.2, name='dropout_3'),
        
        # Output layer
        layers.Dense(n_classes, activation='softmax', name='output')
    ])
    
    return model

# Create model
model = create_model(X_train_scaled.shape[1], n_classes)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model created successfully")
print("\nModel Architecture:")
model.summary()

# ========================
# 4. TRAIN MODEL
# ========================
print("\n[4] Training model...")

# Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

model_checkpoint = callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

print(f"\n✓ Training completed!")
print(f"✓ Model saved to: {MODEL_SAVE_PATH}")

# ========================
# 5. EVALUATE MODEL
# ========================
print("\n[5] Evaluating model...")

# Load best model
model = keras.models.load_model(MODEL_SAVE_PATH)

# Predictions
y_train_pred = np.argmax(model.predict(X_train_scaled, verbose=0), axis=1)
y_val_pred = np.argmax(model.predict(X_val_scaled, verbose=0), axis=1)
y_test_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)

# Calculate accuracies
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)
print(f"Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)")

# Detailed classification report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT (Test Set)")
print("=" * 70)
print(classification_report(
    y_test, 
    y_test_pred, 
    target_names=label_encoder.classes_,
    digits=4
))

# ========================
# 6. VISUALIZATIONS
# ========================
print("\n[6] Generating visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Training History
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/training_history.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {RESULTS_DIR}/training_history.png")
plt.close()

# 2. Confusion Matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

datasets = [
    ('Training', y_train, y_train_pred),
    ('Validation', y_val, y_val_pred),
    ('Test', y_test, y_test_pred)
]

for idx, (name, y_true, y_pred) in enumerate(datasets):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=axes[idx],
        cbar_kws={'label': 'Count'}
    )
    axes[idx].set_title(f'{name} Set Confusion Matrix', fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=12)
    axes[idx].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {RESULTS_DIR}/confusion_matrices.png")
plt.close()

# 3. Feature Importance (using permutation-based approach)
print("\nCalculating feature importance...")
feature_names = X.columns.tolist()
baseline_acc = accuracy_score(y_test, y_test_pred)
importances = []

for i, feature in enumerate(feature_names):
    X_test_permuted = X_test_scaled.copy()
    X_test_permuted[:, i] = np.random.permutation(X_test_permuted[:, i])
    y_pred_permuted = np.argmax(model.predict(X_test_permuted, verbose=0), axis=1)
    permuted_acc = accuracy_score(y_test, y_pred_permuted)
    importance = baseline_acc - permuted_acc
    importances.append(importance)

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance_df)))
ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
ax.set_xlabel('Importance (Accuracy Drop)', fontsize=12)
ax.set_title('Feature Importance Analysis', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/feature_importance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {RESULTS_DIR}/feature_importance.png")
plt.close()

# 4. Class-wise Performance
fig, ax = plt.subplots(figsize=(10, 6))

report_dict = classification_report(
    y_test, 
    y_test_pred, 
    target_names=label_encoder.classes_,
    output_dict=True
)

classes = list(label_encoder.classes_)
metrics = ['precision', 'recall', 'f1-score']
x = np.arange(len(classes))
width = 0.25

for i, metric in enumerate(metrics):
    values = [report_dict[cls][metric] for cls in classes]
    ax.bar(x + i*width, values, width, label=metric.capitalize())

ax.set_xlabel('Behavior Class', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Class-wise Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(classes)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/class_performance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {RESULTS_DIR}/class_performance.png")
plt.close()

# 5. Prediction Distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# True distribution
true_counts = pd.Series(y_test).value_counts().sort_index()
true_labels = [label_encoder.classes_[i] for i in true_counts.index]
axes[0].bar(true_labels, true_counts.values, color=['#2ecc71', '#e74c3c', '#f39c12'])
axes[0].set_title('True Label Distribution (Test Set)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# Predicted distribution
pred_counts = pd.Series(y_test_pred).value_counts().sort_index()
pred_labels = [label_encoder.classes_[i] for i in pred_counts.index]
axes[1].bar(pred_labels, pred_counts.values, color=['#2ecc71', '#e74c3c', '#f39c12'])
axes[1].set_title('Predicted Label Distribution (Test Set)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/prediction_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {RESULTS_DIR}/prediction_distribution.png")
plt.close()

# ========================
# 7. SAVE RESULTS
# ========================
print("\n[7] Saving results...")

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(f'{RESULTS_DIR}/training_history.csv', index=False)
print(f"✓ Saved: {RESULTS_DIR}/training_history.csv")

# Save predictions
predictions_df = pd.DataFrame({
    'True_Label': [label_encoder.classes_[i] for i in y_test],
    'Predicted_Label': [label_encoder.classes_[i] for i in y_test_pred],
    'Correct': y_test == y_test_pred
})
predictions_df.to_csv(f'{RESULTS_DIR}/test_predictions.csv', index=False)
print(f"✓ Saved: {RESULTS_DIR}/test_predictions.csv")

# Save model summary
with open(f'{RESULTS_DIR}/model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print(f"✓ Saved: {RESULTS_DIR}/model_summary.txt")

# Save performance metrics
with open(f'{RESULTS_DIR}/performance_metrics.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("DRIVER BEHAVIOR CLASSIFICATION - PERFORMANCE METRICS\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Training Accuracy:   {train_acc:.4f} ({train_acc*100:.2f}%)\n")
    f.write(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)\n")
    f.write(f"Test Accuracy:       {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
    f.write("=" * 70 + "\n")
    f.write("CLASSIFICATION REPORT (Test Set)\n")
    f.write("=" * 70 + "\n")
    f.write(classification_report(
        y_test, 
        y_test_pred, 
        target_names=label_encoder.classes_,
        digits=4
    ))
print(f"✓ Saved: {RESULTS_DIR}/performance_metrics.txt")

# ========================
# FINAL SUMMARY
# ========================
print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\n✓ Model saved: {MODEL_SAVE_PATH}")
print(f"✓ Results directory: {RESULTS_DIR}/")
print(f"\nFinal Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print("\nGenerated visualizations:")
print(f"  1. training_history.png - Training/validation curves")
print(f"  2. confusion_matrices.png - Confusion matrices for all sets")
print(f"  3. feature_importance.png - Feature importance analysis")
print(f"  4. class_performance.png - Per-class metrics")
print(f"  5. prediction_distribution.png - Label distributions")
print("\n" + "=" * 70)

"""
Train a TensorFlow model to predict congestion_level (0=Free,1=Moderate,2=Heavy).

Inputs: entry_count, exit_count, road_width, time_of_day, weather, avg_speed
Train on CSV: backend/data/traffic_data.csv
Saves:
  - backend/model/traffic_predictor.h5
  - backend/model/scaler.pkl

Usage:
  python train.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from keras import layers, models


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
DATA_CSV = os.path.join(BASE_DIR, 'backend', 'app', 'data', 'traffic_data.csv')
MODEL_DIR = os.path.join('..', 'backend', 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'traffic_predictor.keras')
MODEL_PATH_H5 = os.path.join(MODEL_DIR, 'traffic_predictor.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')


def load_data(csv_path):
    # Read CSV into pandas. Expect columns: entry_count, exit_count, road_width,
    # time_of_day, weather, avg_speed, congestion_level
    df = pd.read_csv(csv_path)
    # Basic validation
    required = ['entry_count', 'exit_count', 'road_width', 'time_of_day', 'weather', 'avg_speed', 'congestion_level']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")
    return df


def preprocess(df):
    # Select features and label
    X = df[['entry_count', 'exit_count', 'road_width', 'time_of_day', 'weather', 'avg_speed']].copy()
    y = df['congestion_level'].astype(int).copy()

    # Fill missing values
    X = X.fillna(0)

    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=['weather'], drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values, scaler



def build_model(input_dim, num_classes=3):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def ensure_model_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    print('Loading data from', DATA_CSV)
    df = load_data(DATA_CSV)

    print('Preprocessing...')
    X, y, scaler = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print('Building model...')
    model = build_model(X.shape[1], num_classes=len(np.unique(y)))
    model.summary()

    print('Training...')
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

    print('Evaluating...')
    preds = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, preds)
    print(f'Accuracy on test set: {acc:.4f}')
    print('\nClassification report:\n', classification_report(y_test, preds))

    print('Saving model and scaler...')
    ensure_model_dir(MODEL_DIR)
    # Save in Keras native format (.keras) as recommended; also save H5 for compatibility
    model.save(MODEL_PATH)
    try:
        model.save(MODEL_PATH_H5)
    except Exception:
        # if H5 save fails (e.g., custom objects), ignore — .keras is preferred
        print('Warning: failed to save H5 file; native .keras file saved')
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print('Saved model to', MODEL_PATH)
    print('Saved scaler to', SCALER_PATH)


if __name__ == '__main__':
    main()

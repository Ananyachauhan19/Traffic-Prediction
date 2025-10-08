
import os
import sqlite3
import time
from typing import Optional

import joblib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import TrafficIngest, PredictionResponse


DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'traffic.db')
DB_PATH = os.path.abspath(DB_PATH)
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, 'traffic_predictor.keras')
MODEL_PATH_H5 = os.path.join(MODEL_DIR, 'traffic_predictor.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

app = FastAPI(title='Traffic Prediction API')
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def init_db():
	conn = sqlite3.connect(DB_PATH)
	c = conn.cursor()
	c.execute('''CREATE TABLE IF NOT EXISTS traffic (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					timestamp INTEGER,
					entry_count INTEGER,
					exit_count INTEGER,
					road_width REAL,
					time_of_day REAL,
					weather REAL,
					avg_speed REAL
				)''')
	conn.commit()
	conn.close()


@app.on_event('startup')
def startup_event():
	init_db()


@app.post('/api/traffic/ingest')
def ingest(payload: TrafficIngest):
	# Store incoming counts into SQLite DB
	ts = int(time.time())
	conn = sqlite3.connect(DB_PATH)
	c = conn.cursor()
	c.execute(
		'INSERT INTO traffic (timestamp, entry_count, exit_count, road_width, time_of_day, weather, avg_speed) VALUES (?, ?, ?, ?, ?, ?, ?)',
		(ts, payload.entry_count, payload.exit_count, payload.road_width, payload.time_of_day, payload.weather, payload.avg_speed),
	)
	conn.commit()
	conn.close()
	return {"status": "ok", "timestamp": ts}


def load_model_and_scaler():
	# Lazy load model/scaler when needed. Using tensorflow inside endpoint to avoid import at module load.
	try:
		import tensorflow as tf
	except Exception:
		raise HTTPException(status_code=500, detail='TensorFlow not available on server')

	# prefer native Keras format, fall back to H5 for compatibility
	if not os.path.exists(SCALER_PATH):
		raise HTTPException(status_code=500, detail='Scaler not found')

	if os.path.exists(MODEL_PATH_KERAS):
		model = tf.keras.models.load_model(MODEL_PATH_KERAS)
	elif os.path.exists(MODEL_PATH_H5):
		model = tf.keras.models.load_model(MODEL_PATH_H5)
	else:
		raise HTTPException(status_code=500, detail='Model file not found (.keras or .h5)')
	scaler = joblib.load(SCALER_PATH)
	return model, scaler


def call_openrouteservice(coords_from, coords_to) -> Optional[dict]:
	key = os.environ.get('OPENROUTESERVICE_KEY')
	if not key:
		return None
	url = 'https://api.openrouteservice.org/v2/directions/driving-car'
	headers = {'Authorization': key, 'Content-Type': 'application/json'}
	body = {
		'coordinates': [coords_from, coords_to]
	}
	try:
		r = requests.post(url, json=body, headers=headers, timeout=5)
		if r.status_code == 200:
			return r.json()
	except Exception:
		return None
	return None


@app.post('/api/traffic/predict')
def predict(payload: TrafficIngest):
	# Load model and scaler
	model, scaler = load_model_and_scaler()

	import numpy as np

	X = np.array([[payload.entry_count, payload.exit_count, payload.road_width, payload.time_of_day, payload.weather, payload.avg_speed]])
	Xs = scaler.transform(X)
	probs = model.predict(Xs)[0]
	pred = int(np.argmax(probs))
	prob = float(probs[pred])

	alternate = None
	if pred >= 2:
		# Example: call OpenRouteService to get alternate route.
		# Need to supply coords (lng, lat). For prototype we'll use placeholders and expect caller to supply real coords later.
		coords_from = [0.0, 0.0]
		coords_to = [0.01, 0.01]
		alt = call_openrouteservice(coords_from, coords_to)
		if alt:
			alternate = alt

	return PredictionResponse(status=str(pred), probability=prob, alternate_route=alternate)

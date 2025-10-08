
"""
edge-interference/detector.py

Detect vehicles using YOLOv8 (ultralytics) from a live or recorded video feed,
count vehicles entering/exiting a defined ROI (virtual line), and POST per-frame
counts to the FastAPI backend endpoint `/api/traffic/ingest`.

Requirements (install in this service env):
  pip install ultralytics opencv-python requests scipy

Usage:
  python detector.py --source 0 --backend http://localhost:8000
  python detector.py --source video.mp4 --backend http://example.com

This script is intentionally lightweight and uses a simple centroid-tracking
approach to determine direction across a counting line.
"""

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import requests

try:
	# ultralytics package provides YOLOv8 model API
	from ultralytics import YOLO
except Exception:
	YOLO = None


@dataclass
class TrackedObject:
	id: int
	centroid: Tuple[int, int]
	positions: deque  # recent centroid positions


class SimpleCentroidTracker:
	"""Very small centroid-based tracker to keep IDs for detections across frames.

	Not meant to replace a full tracker; sufficient for counting across a line.
	"""

	def __init__(self, max_disappeared=10, max_distance=50):
		self.next_object_id = 0
		self.objects: Dict[int, TrackedObject] = {}
		self.disappeared = {}
		self.max_disappeared = max_disappeared
		self.max_distance = max_distance

	def register(self, centroid):
		self.objects[self.next_object_id] = TrackedObject(
			id=self.next_object_id, centroid=centroid, positions=deque([centroid], maxlen=30)
		)
		self.disappeared[self.next_object_id] = 0
		self.next_object_id += 1

	def deregister(self, object_id):
		del self.objects[object_id]
		del self.disappeared[object_id]

	def update(self, input_centroids):
		# If no centroids, mark all as disappeared
		if len(input_centroids) == 0:
			for oid in list(self.disappeared.keys()):
				self.disappeared[oid] += 1
				if self.disappeared[oid] > self.max_disappeared:
					self.deregister(oid)
			return self.objects

		if len(self.objects) == 0:
			for c in input_centroids:
				self.register(c)
			return self.objects

		# Build arrays for distance calculation
		object_ids = list(self.objects.keys())
		object_centroids = [self.objects[oid].centroid for oid in object_ids]
		D = np.linalg.norm(np.array(object_centroids)[:, None] - np.array(input_centroids)[None, :], axis=2)

		rows = D.min(axis=1).argsort()
		cols = D.argmin(axis=1)[rows]

		assigned_cols = set()
		for row, col in zip(rows, cols):
			if col in assigned_cols:
				continue
			if D[row, col] > self.max_distance:
				continue
			oid = object_ids[row]
			c = tuple(map(int, input_centroids[col]))
			self.objects[oid].centroid = c
			self.objects[oid].positions.append(c)
			self.disappeared[oid] = 0
			assigned_cols.add(col)

		# register unassigned input centroids
		for i, c in enumerate(input_centroids):
			if i not in assigned_cols:
				self.register(tuple(map(int, c)))

		# increase disappeared for unassigned object ids
		assigned_rows = {r for r, c in zip(rows, cols) if c in assigned_cols}
		for i, oid in enumerate(object_ids):
			if i not in assigned_rows:
				self.disappeared[oid] += 1
				if self.disappeared[oid] > self.max_disappeared:
					self.deregister(oid)

		return self.objects


def post_counts_async(backend_url: str, payload: dict):
	"""Send the counts to backend in a separate thread to avoid blocking.

	This function is intentionally fire-and-forget but logs on failure.
	"""

	def _worker(url, data):
		try:
			# POST to the ingest endpoint; backend should accept JSON
			r = requests.post(url.rstrip('/') + '/api/traffic/ingest', json=data, timeout=5)
			if r.status_code >= 400:
				print(f"[detector] Failed to POST counts: {r.status_code} {r.text}")
		except Exception as e:
			print(f"[detector] Exception while posting counts: {e}")

	th = threading.Thread(target=_worker, args=(backend_url, payload), daemon=True)
	th.start()


def detect_and_count(source: str, backend: str, yolo_weights: str = 'yolov8n.pt', roi_line=None):
	"""Main routine: open video, run YOLOv8, count vehicles crossing a line.

	Args:
		source: video file path or camera index (string '0' for camera 0)
		backend: base URL of backend service to POST counts to
		yolo_weights: path to YOLOv8 weights (default: 'yolov8n.pt')
		roi_line: ((x1,y1),(x2,y2)) coordinates for counting line in frame coords.
				  If None, a horizontal line at 2/3 of frame height will be used.
	"""

	if YOLO is None:
		raise RuntimeError("ultralytics(YOLOv8) not available. Install with: pip install ultralytics")

	# Load model (uses thread-safe inference in ultralytics, but we still keep it simple)
	model = YOLO(yolo_weights)

	# OpenCV video capture (works with numeric camera index expressed as str)
	try:
		src = int(source)
	except Exception:
		src = source
	cap = cv2.VideoCapture(src)
	if not cap.isOpened():
		raise RuntimeError(f"Cannot open video source: {source}")

	ret, frame = cap.read()
	if not ret:
		raise RuntimeError("Cannot read frame from source")

	h, w = frame.shape[:2]
	if roi_line is None:
		# Default counting line: horizontal line across frame at 2/3 height
		y = int(h * 2 / 3)
		roi_line = ((0, y), (w, y))

	# instantiate tracker
	tracker = SimpleCentroidTracker(max_disappeared=15, max_distance=75)

	# keep track of object IDs that have been counted to avoid double-counting
	counted_ids = set()

	frame_count = 0
	fps_time = time.time()

	# main loop
	while True:
		ret, frame = cap.read()
		if not ret:
			break

		frame_count += 1

		# Run detection (ultralytics will return results with boxes, confidences, class ids)
		# We run with imgsz resized by model internally; use model.predict (or model())
		results = model.predict(frame, imgsz=640, conf=0.3, classes=[2, 3, 5, 7])
		# classes selected: COCO car(2), motorcycle(3), bus(5), truck(7)

		# Parse detections to get centroids
		centroids = []
		boxes = []
		for r in results:
			# r.boxes.xyxy -> tensor of boxes; r.boxes.conf -> confidences; r.boxes.cls -> class ids
			if hasattr(r, 'boxes') and r.boxes is not None:
				for box in r.boxes:
					xyxy = box.xyxy.tolist()[0]
					x1, y1, x2, y2 = map(int, xyxy)
					cx = int((x1 + x2) / 2)
					cy = int((y1 + y2) / 2)
					centroids.append((cx, cy))
					boxes.append((x1, y1, x2, y2))

		# Update tracker with current centroids
		objects = tracker.update(centroids)

		# Initialize per-frame counts
		entry_count = 0
		exit_count = 0

		# Determine if object's trajectory crosses the roi_line between last two positions
		(x1_line, y1_line), (x2_line, y2_line) = roi_line
		for oid, obj in objects.items():
			if len(obj.positions) < 2:
				continue
			prev_x, prev_y = obj.positions[-2]
			cur_x, cur_y = obj.positions[-1]

			# compute line intersection by checking if points are on different sides of the line
			# For a horizontal line, simply compare y coords; generalize with cross product sign
			def side(px, py):
				return np.sign((x2_line - x1_line) * (py - y1_line) - (y2_line - y1_line) * (px - x1_line))

			side_prev = side(prev_x, prev_y)
			side_cur = side(cur_x, cur_y)

			if side_prev == 0 or side_cur == 0 or side_prev == side_cur:
				continue

			# We have a crossing. Determine direction based on dy along the line normal
			# For horizontal line (y increasing downward), moving from top (smaller y) to bottom (larger y)
			# indicates an 'entry' for our ROI below the line; reverse depends on your ROI definition.
			if oid in counted_ids:
				continue

			# Simple direction: compare previous y to current y vs line y
			line_y = y1_line
			if prev_y < line_y and cur_y >= line_y:
				entry_count += 1
				counted_ids.add(oid)
			elif prev_y > line_y and cur_y <= line_y:
				exit_count += 1
				counted_ids.add(oid)

		# Draw visual aids on frame for debugging
		cv2.line(frame, roi_line[0], roi_line[1], (0, 0, 255), 2)
		for oid, obj in objects.items():
			cx, cy = obj.centroid
			cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
			cv2.putText(frame, str(oid), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

		# Show counts on frame
		cv2.putText(frame, f"Entry: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
		cv2.putText(frame, f"Exit: {exit_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

		# Prepare payload and post asynchronously
		payload = {
			"timestamp": int(time.time()),
			"entry_count": entry_count,
			"exit_count": exit_count,
			"frame": frame_count,
		}
		post_counts_async(backend, payload)

		# Show frame (optional) and handle key press
		cv2.imshow('detector', frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

		# Small sleep to yield thread (helps on some systems)
		time.sleep(0.001)

	cap.release()
	cv2.destroyAllWindows()


def parse_args():
	p = argparse.ArgumentParser()
	p.add_argument('--source', type=str, default='0', help='Video source (camera index or file)')
	p.add_argument('--backend', type=str, default='http://localhost:8000', help='Backend base URL')
	p.add_argument('--weights', type=str, default='yolov8n.pt', help='YOLOv8 weights path')
	p.add_argument('--line_y', type=float, default=0.66, help='Relative y position for counting line (0..1)')
	return p.parse_args()


if __name__ == '__main__':
	args = parse_args()
	# convert source '0' to camera index if possible
	detect_and_count(args.source, args.backend, yolo_weights=args.weights, roi_line=None)

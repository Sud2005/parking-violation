# Parking Violation Detection system

A real-time, GPU-accelerated computer vision application for detecting parking violations using YOLOv8, DeepSORT, and OpenCV.

## Overview

This project detects vehicles in a video feed, tracks them using DeepSORT, and monitors how long they stay within a defined Region of Interest (ROI or "No-Parking Zone"). If a tracked vehicle remains in the zone beyond a configurable time threshold, it logs a "violation" and highlights the vehicle.

## Project Structure

- `main.py` - Core execution loop handling video processing, rendering, and logic coordination.
- `detector.py` - Manages the YOLOv8 model for object detection (detects vehicles like cars, trucks, motorcycles).
- `tracker.py` - Implements `deep-sort-realtime` for consistent object tracking across frames.
- `temporal.py` - Monitors continuously how long vehicles stay inside the defined ROI zone.
- `roi.py` & `roi_picker.py` - Tools and methods to define the "No Parking" polygonal zone interactively or synthetically.
- `visualizer.py` - Drawing utilities for bounding boxes, overlay UI, status bars.
- `logger.py` - Saves records of violating vehicles (timestamps, coordinates) to a CSV log.
- `config.yaml` - Main configuration file setting resolutions, threshold limits, models, and more.
- `tests/` - Unit tests for core components.
- `TUNING_GUIDE.md` - Advanced document to fine-tune the tracking and detection confidence.

## Features

- **High-Performance Detection**: Automatically scales down 4k video for efficiency while retaining GPU inferences via YOLOv8.
- **Robust Tracking**: Keeps consistent IDs for objects even through momentary occlusions.
- **Temporal Enforcement**: Calculates actual presence-time and enforces a configurable tolerance limit (e.g., 5 seconds allowed).
- **Comprehensive Logging**: Generates actionable data records exported seamlessly as CSV.

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sud2005/parking-violation.git
   cd parking-violation
   ```

2. **Install dependencies**:
   Ensure you have a Python environment ready (preferably Python 3.9+).
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Settings**:
   Edit `config.yaml` to specify your `video/source`, `violation/time_threshold_seconds`, and paths for the YOLOv8 model.
   
4. **Define your Region of Interest (ROI)**:
   Run the ROI picker utility if you need a fresh polygonal zone.
   ```bash
   python roi_picker.py
   ```
   *Update `config.yaml` with the output points!*

5. **Run the Application**:
   ```bash
   python main.py
   ```

## Requirements

The project primarily uses dependencies configured in `requirements.txt`, such as:
- `ultralytics` (YOLO)
- `opencv-python`
- `deep-sort-realtime`
- `torch` & `torchvision` (optimized for CUDA if available)
- `numpy`, `pyyaml`, `polars`

Wait, wait. It also mentions a `TUNING_GUIDE.md`. Be sure to read it for fine-tuning the params!

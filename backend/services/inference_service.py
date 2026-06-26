# inference_service.py
"""
Thin wrapper around predict.py executing end-to-end inference as a subprocess.
It calls predict.py, parses the prediction.json, and returns a normalized schema.
"""

import time
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, Any


def _normalise_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw pipeline output to a consistent schema.

    Expected raw_result keys (example):
        - asd_probability (float 0‑1)
        - classification (str)
        - model_version (optional)
    """
    asd_prob = float(raw_result.get("asd_probability", 0.0))
    asd_prob = max(0.0, min(1.0, asd_prob))
    td_prob = 1.0 - asd_prob
    label = "ASD" if asd_prob >= 0.5 else "TD"
    confidence = max(asd_prob, td_prob)
    return {
        "prediction_label": label,
        "asd_probability": round(asd_prob, 4),
        "td_probability": round(td_prob, 4),
        "confidence_score": round(confidence, 4),
        "model_version": raw_result.get("model_version", "unknown"),
        "raw_classification": raw_result.get("classification", "unknown"),
    }


def predict(video_path: str) -> Dict[str, Any]:
    """Run the inference pipeline on *video_path* and return a dict.

    Returns keys:
        prediction_label, asd_probability, td_probability, confidence_score,
        processing_time, model_version, raw_classification
    """
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    start = time.time()
    
    # Locate project root (the directory containing predict.py)
    # This file is located in backend/services/
    project_root = Path(__file__).resolve().parent.parent.parent
    predict_script = project_root / "predict.py"
    
    video_name = video_path_obj.stem
    output_dir = project_root / "outputs" / "inference" / video_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(predict_script),
        "--video", str(video_path),
        "--output-dir", str(output_dir)
    ]
    
    # Run the end-to-end inference script
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline failed with exit code {result.returncode}.\n"
            f"Stdout: {result.stdout}\nStderr: {result.stderr}"
        )
        
    pred_json_path = output_dir / "prediction.json"
    if not pred_json_path.is_file():
        raise FileNotFoundError(f"Prediction result file not found: {pred_json_path}")
        
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
        
    if pred_data.get("status") == "failed":
        raise RuntimeError(f"Pipeline failed internal execution: {pred_data.get('error_message')}")
        
    elapsed = round(time.time() - start, 3)
    
    # Map the prediction.json schema to the raw result expected by _normalise_result
    from ..settings import get_config
    # ... existing lines unchanged ...
    raw_output = {
        "asd_probability": pred_data.get("raw_activation", 0.0),
        "classification": pred_data.get("prediction", "unknown"),
        "model_version": get_config().get("modelType", "vgg16+lstm-v1.0")
    }
    
    normalized = _normalise_result(raw_output)
    normalized["processing_time"] = elapsed
    return normalized

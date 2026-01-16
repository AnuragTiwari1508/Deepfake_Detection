import os
import sys
from typing import Any, Dict

import gradio as gr

# Ensure project root is on path so we can import existing inference pipeline
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from inference import inference  # type: ignore


CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")
MODEL_PATH = os.path.join(ROOT_DIR, "checkpoints", "best_model.pth")


def run_inference(video_path: str) -> Dict[str, Any]:
    if not video_path:
        return {"error": "Please upload a video file."}

    if not os.path.exists(MODEL_PATH):
        return {
            "error": "Model checkpoint not found on Space. "
            "Ensure best_model.pth is present in checkpoints/."
        }

    if not os.path.exists(CONFIG_PATH):
        return {
            "error": "config.yaml not found. "
            "Ensure config.yaml is present at the project root."
        }

    try:
        result = inference(video_path, MODEL_PATH, CONFIG_PATH)
        return result
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}


demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Video(label="Deepfake Input Video"),
    outputs=gr.JSON(label="Detection Result"),
    title="Deepfake Detection (Gradio)",
    description=(
        "Upload a facial video to estimate deepfake probability using a dual-stream "
        "RGB + FFT model trained on DFD. The app aggregates predictions over frames "
        "and reports a final REAL/FAKE label with confidence."
    ),
)


if __name__ == "__main__":
    demo.launch()


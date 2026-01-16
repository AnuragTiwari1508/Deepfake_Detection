import argparse
import yaml
import cv2
import torch
import json
import numpy as np
from torchvision import transforms
from models.fusion import DeepFakeDetector
from preprocessing.face_detect import FaceDetector
from preprocessing.fft import FFTProcessor

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def preprocess_face(face_img, fft_processor, transform):
    # face_img is numpy array (224, 224, 3) RGB
    
    # FFT
    fft_feature = fft_processor.process_image(face_img)
    
    # RGB Transform
    rgb_feature = transform(face_img)
    
    return rgb_feature.unsqueeze(0), fft_feature.unsqueeze(0) # Add batch dim

def inference(video_path, model_path, config_path="config.yaml"):
    config = load_config(config_path)
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = DeepFakeDetector(config).to(device)
    if model_path:
        # Load weights
        # Ensure map_location is set to device
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    
    # Preprocessing tools
    detector = FaceDetector(device=device)
    fft_processor = FFTProcessor(size=224)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Process Video
    try:
        faces = detector.process_video(video_path, fps=config['data']['frame_fps'])
    except Exception as e:
        return {"error": f"Face detection failed: {str(e)}"}
    
    if not faces:
        return {"label": "UNKNOWN", "confidence": 0.0, "message": "No faces detected"}
    
    frame_probs = []
    
    with torch.no_grad():
        for face_img in faces:
            rgb, fft = preprocess_face(face_img, fft_processor, transform)
            rgb, fft = rgb.to(device), fft.to(device)
            
            logits, probs = model(rgb, fft)
            frame_probs.append(probs.item())
            
    # Temporal Aggregation
    if not frame_probs:
        return {"label": "UNKNOWN", "confidence": 0.0, "message": "No predictions made"}
        
    avg_prob = np.mean(frame_probs)
    
    label = "FAKE" if avg_prob > config['inference']['threshold'] else "REAL"
    
    result = {
        "label": label,
        "confidence": float(avg_prob),
        "frame_count": len(faces),
        "frame_probs": frame_probs
    }
    
    # Log low confidence
    if 0.4 < avg_prob < 0.6:
        print(f"Low confidence prediction for {video_path}: {avg_prob}")
        
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    result = inference(args.video, args.model, args.config)
    print(json.dumps(result, indent=2))

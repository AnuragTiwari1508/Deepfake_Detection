import argparse
import yaml
import glob
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
from models.fusion import DeepFakeDetector
from train import DeepFakeDataset, get_transforms # Reuse from train.py

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def evaluate(data_dir, model_path, config_path="config.yaml"):
    config = load_config(config_path)
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    
    # Load Data
    real_images = glob.glob(os.path.join(data_dir, "real", "*.*"))
    fake_images = glob.glob(os.path.join(data_dir, "fake", "*.*"))
    
    if not real_images and not fake_images:
        print("No data found.")
        return
        
    all_files = real_images + fake_images
    all_labels = [0] * len(real_images) + [1] * len(fake_images)
    
    dataset = DeepFakeDataset(all_files, all_labels, transform=get_transforms('val'))
    loader = DataLoader(dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    # Load Model
    model = DeepFakeDetector(config).to(device)
    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    
    y_true = []
    y_scores = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for rgb, fft, labels in loader:
            rgb, fft = rgb.to(device), fft.to(device)
            logits, probs = model(rgb, fft)
            
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy().flatten())
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores > 0.5).astype(int)
    
    # Metrics
    auc_score = roc_auc_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    print(f"Results for {data_dir}:")
    print(f"AUC: {auc_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    return {"auc": auc_score, "accuracy": accuracy, "pr_auc": pr_auc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection Evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing 'real' and 'fake' subfolders")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    evaluate(args.data_dir, args.model, args.config)

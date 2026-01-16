import os
import yaml
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
from tqdm import tqdm
import json
import time
import platform

from models.fusion import DeepFakeDetector
from preprocessing.fft import FFTProcessor

# Load Configuration
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Custom Dataset
class DeepFakeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.fft_processor = FFTProcessor(size=224)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Read Image
        # Assuming images are already face crops as per pipeline
        image = cv2.imread(img_path)
        if image is None:
            # Handle error/empty image by returning a zero tensor or skipping
            # For simplicity, we'll return zeros (should be handled in data prep)
            print(f"Warning: Could not read {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # FFT Processing
        fft_feature = self.fft_processor.process_image(image)
        
        # RGB Processing
        if self.transform:
            rgb_feature = self.transform(image)
        else:
            rgb_feature = transforms.ToTensor()(image)

        return rgb_feature, fft_feature, torch.tensor(label, dtype=torch.float32)

def get_transforms(split):
    if split == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(),
            # Note: JPEG compression, Blur, Noise are harder with standard torchvision
            # We can add custom transforms here if needed.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def train(config):
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    # Expecting data/real/ and data/fake/
    real_images = glob.glob(os.path.join("data", "real", "*.*"))
    fake_images = glob.glob(os.path.join("data", "fake", "*.*"))
    
    if not real_images and not fake_images:
        print("No data found in data/real or data/fake. Please populate the data directory.")
        # For demonstration, we won't crash but just warn
    
    all_files = real_images + fake_images
    # Label 0 for Real, 1 for Fake
    all_labels = [0] * len(real_images) + [1] * len(fake_images)
    
    if len(all_files) == 0:
        print("Exiting training due to lack of data.")
        return

    # Video-level split to avoid leakage across frames from the same video
    def base_from_path(p):
        name = os.path.basename(p)
        return name.split('_f')[0]
    # Group bases per class
    real_bases = sorted(list({base_from_path(p) for p in real_images}))
    fake_bases = sorted(list({base_from_path(p) for p in fake_images}))
    val_ratio = float(config['data']['val_split'])
    test_ratio = float(config['data']['test_split'])
    temp_ratio = val_ratio + test_ratio if (val_ratio + test_ratio) > 0 else 0.0
    def split_bases(bases):
        if temp_ratio > 0:
            train_b, temp_b = train_test_split(bases, test_size=temp_ratio, random_state=42)
            if test_ratio > 0:
                # Split temp into val and test by proportion
                val_share = val_ratio / temp_ratio if temp_ratio > 0 else 0.0
                val_b, test_b = train_test_split(temp_b, test_size=1.0 - val_share, random_state=42)
            else:
                val_b, test_b = temp_b, []
        else:
            train_b, val_b, test_b = bases, [], []
        return set(train_b), set(val_b), set(test_b)
    real_train_b, real_val_b, real_test_b = split_bases(real_bases)
    fake_train_b, fake_val_b, fake_test_b = split_bases(fake_bases)
    def filter_by_bases(images, allowed_bases):
        out = []
        for p in images:
            if base_from_path(p) in allowed_bases:
                out.append(p)
        return out
    # Build image lists per split
    X_train_real = filter_by_bases(real_images, real_train_b)
    X_val_real   = filter_by_bases(real_images, real_val_b)
    X_train_fake = filter_by_bases(fake_images, fake_train_b)
    X_val_fake   = filter_by_bases(fake_images, fake_val_b)
    X_train = X_train_real + X_train_fake
    y_train = [0] * len(X_train_real) + [1] * len(X_train_fake)
    X_val = X_val_real + X_val_fake
    y_val = [0] * len(X_val_real) + [1] * len(X_val_fake)
    class_counts = {0: len(X_train_real), 1: len(X_train_fake)}

    logs_root = os.path.join(config['train']['save_dir'], "logs")
    os.makedirs(logs_root, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    meta_path = os.path.join(logs_root, f"meta_{run_id}.json")
    metrics_path = os.path.join(logs_root, f"metrics_{run_id}.jsonl")
    meta = {
        "run_id": run_id,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "system": platform.platform(),
        "python_version": platform.python_version(),
        "config": config,
        "dataset": {
            "train": {"real_images": len(X_train_real), "fake_images": len(X_train_fake),
                      "real_videos": len(real_train_b), "fake_videos": len(fake_train_b)},
            "val": {"real_images": len(X_val_real), "fake_images": len(X_val_fake),
                    "real_videos": len(real_val_b), "fake_videos": len(fake_val_b)}
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    train_dataset = DeepFakeDataset(X_train, y_train, transform=get_transforms('train'))
    val_dataset = DeepFakeDataset(X_val, y_val, transform=get_transforms('val'))

    sample_weights = [1.0 / class_counts[label] for label in y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'], 
        sampler=sampler, 
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=False, 
        num_workers=config['data']['num_workers']
    )

    # 2. Model
    model = DeepFakeDetector(config).to(device)

    # 3. Optimizer & Loss
    # Separate learning rates
    head_params = list(model.classifier.parameters()) + list(model.fft_branch.parameters()) # FFT branch is custom trained from scratch? Yes.
    # Actually user says: "Freeze backbone for first N epochs".
    # And "LR: 1e-4 (head), 1e-5 (backbones)"
    
    # We should handle freezing logic in the loop or setup groups
    optimizer = optim.AdamW([
        {'params': model.rgb_branch.parameters(), 'lr': float(config['train']['learning_rate_backbone'])},
        {'params': model.fft_branch.parameters(), 'lr': float(config['train']['learning_rate_head'])}, # FFT is custom, maybe higher LR? User said 1e-4 (head), 1e-5 (backbones). FFT is custom so maybe head LR.
        {'params': model.classifier.parameters(), 'lr': float(config['train']['learning_rate_head'])}
    ], weight_decay=float(config['train']['weight_decay']))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])
    pos_weight = torch.tensor(class_counts[0] / max(1, class_counts[1]), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Training Loop
    best_auc = 0.0
    start_epoch = 0
    os.makedirs(config['train']['save_dir'], exist_ok=True)
    last_ckpt = os.path.join(config['train']['save_dir'], "checkpoint_last.pth")
    if bool(config['train'].get('resume', False)) and os.path.exists(last_ckpt):
        state = torch.load(last_ckpt, map_location=device)
        try:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            best_auc = state.get("best_auc", 0.0)
            start_epoch = state.get("epoch", 0) + 1
            print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"Resume failed: {e}")

    for epoch in range(start_epoch, config['train']['epochs']):
        model.train()
        
        # Freeze backbone for first N epochs? User said "Freeze backbone for first N epochs".
        # Let's say N=5 for example, or configurable.
        # We didn't put N in config, let's assume N=3.
        if epoch < 3:
             model.rgb_branch.set_trainable(False)
        else:
             model.rgb_branch.set_trainable(True)

        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        for rgb, fft, labels in pbar:
            if os.path.exists(config['train'].get('pause_file', "")):
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_auc": best_auc,
                    "run_id": run_id
                }, os.path.join(config['train']['save_dir'], "checkpoint_paused.pth"))
                print("Paused training and saved checkpoint.")
                return
            rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, probs = model(rgb, fft)
            logits_flat = logits.view(-1)
            labels_flat = labels.view(-1)
            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for rgb, fft, labels in val_loader:
                rgb, fft, labels = rgb.to(device), fft.to(device), labels.to(device)
                logits, probs = model(rgb, fft)
                val_probs.extend(probs.cpu().numpy().flatten())
                val_labels.extend(labels.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_probs)
        val_acc = accuracy_score(val_labels, np.array(val_probs) > 0.5)
        precision, recall, _ = precision_recall_curve(val_labels, val_probs)
        pr_auc = auc(recall, precision)
        print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f} - Val AUC: {val_auc:.4f} - Val Acc: {val_acc:.4f} - PR-AUC: {pr_auc:.4f}")
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        metrics_entry = {
            "epoch": epoch + 1,
            "train_loss": running_loss/len(train_loader),
            "val_auc": float(val_auc),
            "val_accuracy": float(val_acc),
            "val_pr_auc": float(pr_auc),
            "lrs": lrs,
            "timestamp": time.time()
        }
        with open(metrics_path, "a") as mf:
            mf.write(json.dumps(metrics_entry) + "\n")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_auc": best_auc,
            "run_id": run_id
        }, last_ckpt)
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], "best_model.pth"))
            print("Saved best model.")

if __name__ == "__main__":
    config = load_config()
    train(config)

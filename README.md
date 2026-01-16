# Deepfake Detection System

A robust deep learning-based system for detecting manipulated facial videos (Deepfakes). This project utilizes a dual-stream architecture combining spatial features (RGB via EfficientNet) and frequency domain features (FFT) to accurately identify fake content.

## 🚀 Key Features

- **Dual-Stream Architecture**: 
  - **RGB Stream**: Uses `EfficientNet-B0` to capture visual artifacts and spatial inconsistencies.
  - **Frequency Stream**: Uses Fast Fourier Transform (FFT) to detect spectral anomalies common in GAN-generated images.
- **Video-Aware Data Management**:
  - **Leakage Prevention**: Ensures frames from the same video never cross between Training, Validation, and Test sets.
  - **Physical & Logical Balancing**: Handles severe class imbalance (e.g., 65k Real vs 35k Fake) via both physical dataset curation and Weighted Random Sampling during training.
- **Robust Training Pipeline**:
  - **Class-Weighted Loss**: Mitigates bias towards majority classes.
  - **Pause/Resume Support**: Automatically saves and loads checkpoints, allowing training to be paused and resumed without progress loss.
  - **Detailed Logging**: Tracks metrics like AUC, Precision, Recall, and Loss per epoch.

## 📂 Project Structure

```
Deepfake_Detection/
├── config.yaml                 # Central configuration for Data, Model, and Training
├── train.py                    # Main training script
├── inference.py                # Script for running inference on single videos/images
├── evaluate.py                 # Script for evaluating model on test set
├── requirements.txt            # Python dependencies
├── preprocessing/
│   ├── build_dfd_dataset.py    # Extracts faces and builds balanced datasets
│   ├── face_detect.py          # MTCNN-based face detection wrapper
│   └── fft.py                  # FFT feature extraction logic
├── models/
│   └── fusion.py               # Dual-stream model definition
└── checkpoints/                # Saved models and training states
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sahu-Chitransh/Deepfake_Detection.git
   cd Deepfake_Detection
   ```

2. **Create a virtual environment (Optional but Recommended)**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Preparation

The system works by extracting face crops from raw video datasets.

1. **Configure paths** in `preprocessing/build_dfd_dataset.py` (or ensure your raw data is in `raw_data/`).
2. **Run the builder**:
   ```bash
   python preprocessing/build_dfd_dataset.py
   ```
   *This script will:*
   - Detect and extract faces from videos.
   - Balance the dataset (downsampling majority class if needed).
   - Organize data into `data/real` and `data/fake`.

## 🧠 Training

To start training the model:

```bash
python train.py
```

- **Configuration**: Modify `config.yaml` to adjust hyperparameters like `batch_size`, `learning_rate`, `epochs`, etc.
- **Checkpoints**: Models are saved to `./checkpoints/`. The script automatically resumes from the latest checkpoint if interrupted.
- **Pause Training**: Create a file named `PAUSE` in the `./checkpoints/` directory to safely stop training after the current epoch.

## 🔍 Inference

To run detection on a specific image or video:

```bash
python inference.py --input path/to/video.mp4
```
*(Note: Ensure `inference.py` is configured to load the best checkpoint)*

## ⚙️ Configuration (`config.yaml`)

| Section | Key | Description |
| :--- | :--- | :--- |
| **Data** | `frame_fps` | Frames per second to extract from videos |
| | `image_size` | Input resolution (default: 224x224) |
| **Model** | `rgb_backbone` | Backbone network (e.g., efficientnet-b0) |
| | `fft_feature_dim` | Dimension of frequency features |
| **Train** | `label_smoothing` | Regularization technique for soft labels |
| | `save_dir` | Directory for saving model weights |

## 📜 License

This project is open-source. Please check the repository for license details.

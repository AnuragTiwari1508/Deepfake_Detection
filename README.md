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
├── inference.py                # Script for running inference on videos
├── evaluate.py                 # Script for evaluating model on test set
├── requirements.txt            # Python dependencies (includes FastAPI and Uvicorn)
├── preprocessing/
│   ├── build_dfd_dataset.py    # Extracts faces and builds balanced datasets
│   ├── face_detect.py          # MTCNN-based face detection wrapper
│   └── fft.py                  # FFT feature extraction logic
├── models/
│   └── fusion.py               # Dual-stream model definition
├── backend/
│   └── main.py                 # FastAPI service exposing /detect endpoint
├── frontend/
│   └── index.html              # Vue-based single-page UI for video upload and insights
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

## 🔍 Inference (CLI)

To run detection on a specific video directly from Python:

```bash
python inference.py --video path/to/video.mp4 --model checkpoints/best_model.pth
```

This prints a JSON blob like:

```json
{
  "label": "FAKE",
  "confidence": 0.87,
  "frame_count": 42,
  "frame_probs": [0.81, 0.85, 0.89, ...]
}
```

Where:
- `label` is the final decision (REAL / FAKE / UNKNOWN).
- `confidence` is the aggregated deepfake probability across frames.
- `frame_probs` are per-frame probabilities used for more detailed analysis.

## 🌐 Backend API (FastAPI)

For serving the model over HTTP, a FastAPI service wraps `inference.py`.

From the project root:

```bash
.\.venv\Scripts\activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /`  
  Health check and model status:
  ```json
  {
    "status": "Deepfake Detection API is running",
    "model_loaded": true
  }
  ```

- `POST /detect`  
  - Request: `multipart/form-data` with field `file` (video: mp4, avi, mov, mkv, webm).
  - Response (example):
    ```json
    {
      "label": "FAKE",
      "confidence": 0.8731,
      "frame_count": 42,
      "frame_probs": [0.81, 0.85, 0.89, ...]
    }
    ```

The backend automatically selects the best checkpoint from `checkpoints/` (prefers `best_model.pth`, otherwise latest `*.pth`).

## 💻 Web UI (Vue Frontend)

A modern single-page UI is provided in `frontend/index.html`. It lets you:

- Upload a video for deepfake detection.
- See a rich result view with:
  - Final prediction label and confidence.
  - Frames analyzed.
  - Prediction stability (based on frame-level variance).
  - Confidence bar and a mini bar chart over sampled frames.
- Experience a full-screen loading overlay while inference runs.

### Running the dashboard locally

1. Start the backend (from project root):
   ```bash
   .\.venv\Scripts\activate
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

2. Serve the frontend (optional, but recommended instead of opening the file directly):
   ```bash
   cd frontend
   python -m http.server 5173
   ```

3. Open the UI in your browser:
   ```text
   http://localhost:5173
   ```

The frontend connects to the backend at `http://localhost:8000/detect` by default.

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

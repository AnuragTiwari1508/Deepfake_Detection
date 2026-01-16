import sys
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import inference

app = FastAPI(title="Deepfake Detection API", description="API for detecting deepfake videos")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

def get_best_model_path():
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if not files:
        return None
        
    if "best_model.pth" in files:
        return os.path.join(CHECKPOINT_DIR, "best_model.pth")
    
    # Return the latest modified file
    files = [os.path.join(CHECKPOINT_DIR, f) for f in files]
    return max(files, key=os.path.getmtime)

@app.get("/")
def read_root():
    return {"status": "Deepfake Detection API is running", "model_loaded": get_best_model_path() is not None}

@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    # Validate file extension
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    # Save file temporarily
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        model_path = get_best_model_path()
        if not model_path:
            raise HTTPException(status_code=500, detail="No model checkpoint found. Please train the model first.")
            
        # Run inference
        result = inference(file_path, model_path)
        
        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
             
        return result
        
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import cv2
import torch
import numpy as np
from PIL import Image
try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("facenet_pytorch not installed. Please install it using: pip install facenet-pytorch")
    MTCNN = None

class FaceDetector:
    def __init__(self, device='cuda', image_size=224, margin=0):
        self.device = device
        if MTCNN is not None:
            self.mtcnn = MTCNN(
                image_size=image_size, 
                margin=margin, 
                keep_all=True, # We'll select the largest ourselves
                device=device,
                post_process=False # We want the raw image, not normalized
            )
        else:
            self.mtcnn = None
            
    def process_video(self, video_path, fps=5):
        """
        Extract faces from video.
        Returns a list of (face_rgb, frame_idx) tuples.
        """
        if self.mtcnn is None:
            raise ImportError("MTCNN not initialized.")

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(max(1, round(video_fps / fps)))
        
        faces = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Detect faces
                boxes, _ = self.mtcnn.detect(pil_img)
                
                if boxes is not None and len(boxes) > 0:
                    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                    largest_idx = np.argmax(areas)
                    box = boxes[largest_idx]
                    b = [int(max(0, c)) for c in box]
                    face_img = frame_rgb[b[1]:b[3], b[0]:b[2]]
                    if face_img.size > 0:
                        face_img = cv2.resize(face_img, (224, 224))
                        faces.append(face_img)
            
            frame_idx += 1
            
        cap.release()
        return faces

    def process_image(self, image_path):
        """
        Process a single image path.
        """
        if self.mtcnn is None:
            raise ImportError("MTCNN not initialized.")
            
        frame = cv2.imread(image_path)
        if frame is None:
            return None
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        boxes, _ = self.mtcnn.detect(pil_img)
        
        if boxes is not None and len(boxes) > 0:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            largest_idx = np.argmax(areas)
            box = boxes[largest_idx]
            
            b = [int(max(0, c)) for c in box]
            face_img = frame_rgb[b[1]:b[3], b[0]:b[2]]
            
            if face_img.size > 0:
                face_img = cv2.resize(face_img, (224, 224))
                return face_img
        
        return None

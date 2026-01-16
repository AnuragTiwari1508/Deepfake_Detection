import cv2
import numpy as np
import torch
from torchvision import transforms

class FFTProcessor:
    def __init__(self, size=224):
        self.size = size

    def process_image(self, image):
        """
        Process a single image (numpy array, RGB) into its FFT representation.
        
        Args:
            image: numpy array of shape (H, W, 3) in RGB format.
            
        Returns:
            fft_feature: torch tensor of shape (1, size, size)
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 2. Resize to target size (if not already) - strictly speaking specs say resize at end, 
        # but FFT on 224x224 is standard. However, the specs say "Resize to 224x224x1" at the end.
        # Let's perform FFT on the resized image to ensure consistent frequency resolution.
        gray = cv2.resize(gray, (self.size, self.size))
        
        # 3. Apply FFT
        f = np.fft.fft2(gray)
        
        # 4. Shift zero-frequency to center
        fshift = np.fft.fftshift(f)
        
        # 5. Take log magnitude spectrum
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8) # Add epsilon to avoid log(0)
        
        # 6. Normalize to [0,1]
        # Normalize based on min/max of the current image or global stats?
        # Usually per-image normalization is robust for this task.
        m_min = np.min(magnitude_spectrum)
        m_max = np.max(magnitude_spectrum)
        
        if m_max - m_min > 1e-8:
            norm_spectrum = (magnitude_spectrum - m_min) / (m_max - m_min)
        else:
            norm_spectrum = np.zeros_like(magnitude_spectrum)
            
        # 7. Convert to tensor and add channel dimension
        # Output: 224x224x1 -> (1, 224, 224) for PyTorch
        fft_feature = torch.from_numpy(norm_spectrum).float().unsqueeze(0)
        
        return fft_feature

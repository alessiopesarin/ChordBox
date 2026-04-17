import os
import torch
import librosa
import numpy as np
import glob
from tqdm import tqdm

# --- CONFIGURATION ---
# Folder where the downloaded WAV/MP3 files are stored
AUDIO_DIR = "data/raw/youtube_audio"

# Folder where tensors ready for the GPU will be saved
OUTPUT_DIR = "data/processed/youtube"

SR = 22050
HOP_LENGTH = 512

def precompute():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🔍 Scanning directory: {AUDIO_DIR}")
    # Support both mp3 and wav files
    audio_files = glob.glob(os.path.join(AUDIO_DIR, "*.mp3")) + glob.glob(os.path.join(AUDIO_DIR, "*.wav"))

    print(f"Found {len(audio_files)} tracks. Starting CQT extraction and normalization...")

    for audio_path in tqdm(audio_files, desc="Precomputing"):
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{filename}.pt")
        
        if os.path.exists(out_path):
            continue 
            
        try:
            # 1. Load audio
            y, sr = librosa.load(audio_path, sr=SR, mono=True)
            
            # 2. Extract CQT
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH).T
            
            # 3. CRITICAL: Normalization identical to Billboard/Inference
            chroma = (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-8)
            
            # 4. Convert to Tensor [Time, 12]
            chroma_tensor = torch.tensor(chroma, dtype=torch.float32)
            
            # 5. Save
            torch.save(chroma_tensor, out_path)
            
        except Exception as e:
            print(f"\n❌ Error processing {filename}: {e}")

    print("\n🎉 YouTube Dataset pre-computing completed!")

if __name__ == "__main__":
    precompute()

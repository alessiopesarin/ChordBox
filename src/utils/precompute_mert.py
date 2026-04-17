import os
import torch
import librosa
import numpy as np
import glob
from transformers import Wav2Vec2FeatureExtractor, AutoModel

# --- CONFIGURATION ---
MODEL_NAME = "m-a-p/MERT-v1-95M"
TARGET_SR = 24000 # MERT-v1-95M requires 24kHz
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping our datasets to their raw audio paths
DATASETS = {
    "billboard": {
        "audio_dir": "data/raw/billboard/audio",
        "output_dir": "data/processed/mert/billboard"
    },
    "guitarset": {
        "audio_dir": "data/raw/guitarset/audio_mono-mic", # Using mono-mic for features
        "output_dir": "data/processed/mert/guitarset"
    }
}

def precompute_dataset(name, config, model, processor):
    audio_files = glob.glob(os.path.join(config["audio_dir"], "*.wav")) + \
                  glob.glob(os.path.join(config["audio_dir"], "*.mp3"))
    
    os.makedirs(config["output_dir"], exist_ok=True)
    print(f"\n🔍 Processing dataset: {name.upper()} ({len(audio_files)} files)")

    # 10 seconds chunks to be safe with 6GB VRAM
    CHUNK_SEC = 10 
    CHUNK_SAMPLES = CHUNK_SEC * TARGET_SR

    with torch.no_grad():
        for i, audio_path in enumerate(audio_files):
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            out_path = os.path.join(config["output_dir"], f"{filename}.pt")
            
            if os.path.exists(out_path):
                continue 
                
            try:
                # 1. Load at 24kHz
                y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
                
                # 2. Process in chunks
                all_hidden_states = []
                for start in range(0, len(y), CHUNK_SAMPLES):
                    end = min(start + CHUNK_SAMPLES, len(y))
                    chunk = y[start:end]
                    
                    if len(chunk) < 1600: # Skip extremely short tails
                        continue

                    inputs = processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt")
                    input_values = inputs.input_values.to(DEVICE)
                    
                    outputs = model(input_values)
                    # Shape: [1, chunk_frames, 768]
                    hidden_states = outputs.last_hidden_state.squeeze(0).cpu()
                    all_hidden_states.append(hidden_states)
                    
                    # Clean up VRAM immediately
                    del outputs, input_values
                    torch.cuda.empty_cache()

                # 3. Concatenate and Save
                if all_hidden_states:
                    final_tensor = torch.cat(all_hidden_states, dim=0)
                    torch.save(final_tensor, out_path)
                    
                    if (i+1) % 10 == 0 or i == 0:
                        print(f"[{i+1}/{len(audio_files)}] ✅ Saved: {filename}.pt (Shape: {final_tensor.shape})")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                torch.cuda.empty_cache()

def main():
    print(f"⏳ Loading MERT model ({MODEL_NAME}) on {DEVICE}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
    model.eval()

    for ds_name, ds_config in DATASETS.items():
        precompute_dataset(ds_name, ds_config, model, processor)

    print("\n🎉 Music2Vec pre-computing finished!")

if __name__ == "__main__":
    main()

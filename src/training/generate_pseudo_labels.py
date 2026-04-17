import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import glob
import re
from tqdm import tqdm

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.enhanced_crnn import EnhancedChordCRNN
from src.models.deep_crnn import DeepChordCRNN
from src.utils.config_loader import load_config, Config

# --- CONFIGURATION ---
YOUTUBE_FEATURES_DIR = "data/processed/youtube"
OUTPUT_CSV = "data/processed/youtube/youtube_pseudo_labels.csv"
MODEL_DIR = "./models"

CHORDS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
          'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm', 'N.C.']

def get_historical_best_model(model_dir, base_name):
    """Finds the model with the lowest loss for a given prefix."""
    pattern = rf"^{re.escape(base_name)}_([0-9]+\.[0-9]+)\.pth$"
    best_loss = float('inf')
    best_file = None
    if not os.path.exists(model_dir): return None, None
    for f in os.listdir(model_dir):
        match = re.match(pattern, f)
        if match:
            loss = float(match.group(1))
            if loss < best_loss:
                best_loss = loss
                best_file = os.path.join(model_dir, f)
    return best_loss, best_file

def main():
    parser = argparse.ArgumentParser(description="Ensemble Pseudo-Label Generation (Chunked Edition)")
    parser.add_argument("--teacher_enh", type=str, default="student_enhanced_gen_3", help="Base name of the Enhanced model")
    parser.add_argument("--teacher_deep", type=str, default="deep", help="Base name of the Deep model")
    parser.add_argument("--threshold", type=float, default=0.75, help="Minimum confidence threshold")
    parser.add_argument("--chunk_size", type=int, default=430, help="Chunk size to avoid OOM")
    args = parser.parse_args()

    config_dict = load_config("config/default.yaml")
    cfg = Config(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🧠 --- ENSEMBLE CHUNKED LABELING SYSTEM ---")
    print(f"Threshold: {args.threshold} | Chunk Size: {args.chunk_size}")
    
    # 1. Loading ENHANCED model
    model_enh = EnhancedChordCRNN(
        num_classes=25, 
        lstm_hidden=cfg.model.lstm_hidden, 
        lstm_layers=cfg.model.lstm_layers, 
        attention_heads=cfg.model.attention_heads
    ).to(device)
    
    _, path_enh = get_historical_best_model(MODEL_DIR, args.teacher_enh)
    if path_enh:
        print(f"✅ Enhanced Teacher: {os.path.basename(path_enh)}")
        model_enh.load_state_dict(torch.load(path_enh, map_location=device, weights_only=True))
        model_enh.eval()
    else:
        print(f"❌ Error: Enhanced '{args.teacher_enh}' not found.")
        return

    # 2. Loading DEEP model
    model_deep = DeepChordCRNN(num_classes=25).to(device)
    _, path_deep = get_historical_best_model(MODEL_DIR, args.teacher_deep)
    if path_deep:
        print(f"✅ Deep Teacher: {os.path.basename(path_deep)}")
        model_deep.load_state_dict(torch.load(path_deep, map_location=device, weights_only=True))
        model_deep.eval()
    else:
        print(f"❌ Error: Deep '{args.teacher_deep}' not found.")
        return

    # 3. Processing YouTube tracks
    pt_files = glob.glob(os.path.join(YOUTUBE_FEATURES_DIR, "*.pt"))
    dataset_records = []

    with torch.no_grad():
        for pt_path in tqdm(pt_files, desc="Processing tracks"):
            filename = os.path.basename(pt_path)
            try:
                full_chroma = torch.load(pt_path, weights_only=True)
                if full_chroma.dim() == 2: full_chroma = full_chroma.unsqueeze(0).unsqueeze(0)
                elif full_chroma.dim() == 3: full_chroma = full_chroma.unsqueeze(1)
                
                total_frames = full_chroma.shape[2]
                all_probs_enh = []
                all_probs_deep = []

                # --- CHUNKED INFERENCE ---
                for start_idx in range(0, total_frames, args.chunk_size):
                    end_idx = min(start_idx + args.chunk_size, total_frames)
                    chunk = full_chroma[:, :, start_idx:end_idx, :].to(device)
                    
                    if chunk.shape[2] < 5: continue

                    p_enh = torch.softmax(model_enh(chunk), dim=-1).squeeze(0).cpu()
                    p_deep = torch.softmax(model_deep(chunk), dim=-1).squeeze(0).cpu()
                    
                    all_probs_enh.append(p_enh)
                    all_probs_deep.append(p_deep)
                
                # Merge processed chunks
                if not all_probs_enh: continue
                probs_enh = torch.cat(all_probs_enh, dim=0)
                probs_deep = torch.cat(all_probs_deep, dim=0)
                
                # Ensemble logic (Executed on CPU to save VRAM)
                for t in range(probs_enh.shape[0]):
                    conf_enh, pred_enh = torch.max(probs_enh[t], dim=-1)
                    conf_deep, pred_deep = torch.max(probs_deep[t], dim=-1)
                    
                    if conf_enh >= conf_deep:
                        best_conf = conf_enh.item()
                        best_pred = pred_enh.item()
                        source = "enh"
                    else:
                        best_conf = conf_deep.item()
                        best_pred = pred_deep.item()
                        source = "deep"
                        
                    if best_conf >= args.threshold:
                        dataset_records.append({
                            "filename": filename,
                            "frame_idx": t,
                            "chord": CHORDS[best_pred],
                            "confidence": best_conf,
                            "teacher": source
                        })
                
                # Clear CUDA cache after each track for safety
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n⚠️ Error on {filename}: {e}")

    # 4. Saving
    if dataset_records:
        df = pd.DataFrame(dataset_records)
        df.to_csv(OUTPUT_CSV, index=False)
        enh_count = len(df[df['teacher'] == 'enh'])
        deep_count = len(df[df['teacher'] == 'deep'])
        print(f"\n🎉 Ensemble complete! {len(df)} pseudo-labels generated.")
        print(f"📊 Stats: Enhanced {enh_count} | Deep {deep_count}")
    else:
        print("\n⚠️ No labels found.")

if __name__ == "__main__":
    main()

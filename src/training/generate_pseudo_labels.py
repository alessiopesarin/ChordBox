import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import glob
import re
import random
from tqdm import tqdm

# Aggiunge la root del progetto al PYTHONPATH
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
    parser = argparse.ArgumentParser(description="Ensemble Pseudo-Label Generation (Smart Stochastic)")
    parser.add_argument("--teacher_enh", type=str, default="student_enhanced_gen_6", help="Enhanced teacher")
    parser.add_argument("--teacher_deep", type=str, default="super_master_deep_balanced", help="Deep teacher")
    parser.add_argument("--threshold", type=float, default=0.80, help="Confidence threshold")
    parser.add_argument("--chunk_size", type=int, default=430, help="Chunk size for OOM fix")
    parser.add_argument("--max_tracks", type=int, default=None, help="Limit processing to a random subset of tracks")
    args = parser.parse_args()

    config_dict = load_config("config/default.yaml")
    cfg = Config(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🧠 --- SMART STOCHASTIC LABELING ---")
    
    # 1. Caricamento Modelli
    model_enh = EnhancedChordCRNN(num_classes=25, lstm_hidden=cfg.model.lstm_hidden, 
                                  lstm_layers=cfg.model.lstm_layers, attention_heads=cfg.model.attention_heads).to(device)
    _, path_enh = get_historical_best_model(MODEL_DIR, args.teacher_enh)
    
    model_deep = DeepChordCRNN(num_classes=25).to(device)
    _, path_deep = get_historical_best_model(MODEL_DIR, args.teacher_deep)

    if not path_enh or not path_deep:
        print("❌ Error: Missing models.")
        return
        
    model_enh.load_state_dict(torch.load(path_enh, map_location=device, weights_only=True))
    model_deep.load_state_dict(torch.load(path_deep, map_location=device, weights_only=True))
    model_enh.eval()
    model_deep.eval()

    # 2. Selezione Brani (Stochastic)
    pt_files = glob.glob(os.path.join(YOUTUBE_FEATURES_DIR, "*.pt"))
    if args.max_tracks and args.max_tracks < len(pt_files):
        random.shuffle(pt_files)
        pt_files = pt_files[:args.max_tracks]
        print(f"🎲 Randomly sampled {len(pt_files)} tracks for this generation.")
    else:
        print(f"📂 Processing all {len(pt_files)} tracks.")

    dataset_records = []

    with torch.no_grad():
        for pt_path in tqdm(pt_files, desc="Labeling"):
            filename = os.path.basename(pt_path)
            try:
                full_chroma = torch.load(pt_path, weights_only=True)
                if full_chroma.dim() == 2: full_chroma = full_chroma.unsqueeze(0).unsqueeze(0)
                elif full_chroma.dim() == 3: full_chroma = full_chroma.unsqueeze(1)
                
                total_frames = full_chroma.shape[2]
                all_probs_enh = []
                all_probs_deep = []

                for start_idx in range(0, total_frames, args.chunk_size):
                    end_idx = min(start_idx + args.chunk_size, total_frames)
                    chunk = full_chroma[:, :, start_idx:end_idx, :].to(device)
                    if chunk.shape[2] < 5: continue

                    all_probs_enh.append(torch.softmax(model_enh(chunk), dim=-1).squeeze(0).cpu())
                    all_probs_deep.append(torch.softmax(model_deep(chunk), dim=-1).squeeze(0).cpu())
                
                if not all_probs_enh: continue
                p_enh = torch.cat(all_probs_enh, dim=0)
                p_deep = torch.cat(all_probs_deep, dim=0)
                
                for t in range(p_enh.shape[0]):
                    c_enh, pr_enh = torch.max(p_enh[t], dim=-1)
                    c_deep, pr_deep = torch.max(p_deep[t], dim=-1)
                    
                    # Ensemble logic
                    if c_enh >= c_deep:
                        best_c, best_p, src = c_enh.item(), pr_enh.item(), "enh"
                    else:
                        best_c, best_p, src = c_deep.item(), pr_deep.item(), "deep"
                        
                    if best_c >= args.threshold:
                        dataset_records.append({"filename": filename, "frame_idx": t, "chord": CHORDS[best_p], 
                                                "confidence": best_c, "teacher": src})
                
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️ Error {filename}: {e}")

    if dataset_records:
        pd.DataFrame(dataset_records).to_csv(OUTPUT_CSV, index=False)
        print(f"🎉 Generated {len(dataset_records)} pseudo-labels.")
    else:
        print("⚠️ No labels generated.")

if __name__ == "__main__":
    main()

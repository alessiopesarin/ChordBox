import torch
import torch.nn as nn
import pandas as pd
import os
import glob
import random
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

# Import architectures and datasets
from src.models.enhanced_crnn import EnhancedChordCRNN
from src.data.datasets import BillboardDataset, PseudoLabelDataset, GuitarSetDirectDataset, pad_collate_fn
from src.training.trainer import ChordTrainer
from src.utils.config_loader import load_config, Config

def get_historical_best_model(model_dir, base_name):
    """Dynamically finds the best model file based on the prefix."""
    import re
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
    parser = argparse.ArgumentParser(description="Train a Student model using Hybrid Ensemble Labels.")
    parser.add_argument("--teacher", type=str, default="student_enhanced_gen_3", help="Enhanced model for weight initialization")
    parser.add_argument("--student_name", type=str, default="student_enhanced_gen_5", help="Name for the new student generation")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    args = parser.parse_args()

    config_dict = load_config("config/default.yaml")
    cfg = Config(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🚀 STARTING {args.student_name.upper()} (Hybrid Ensemble Noisy Student)")
    print(f"Device: {device} | Using 70/30 Gold-to-Silver anchor ratio.")

    # --- 1. GOLD DATA (Billboard + GuitarSet) ---
    full_df = pd.read_csv(cfg.data.metadata_csv, dtype={'Track_ID': str})
    valid_b = [row for _, row in full_df.iterrows() if os.path.exists(os.path.join(cfg.data.audio_dir, f"{row['Track_ID']}.wav"))]
    valid_b_df = pd.DataFrame(valid_b).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # GuitarSet
    g_anno_dir = cfg.data.guitarset_annotations_dir
    g_pt_files = glob.glob(os.path.join(cfg.data.guitarset_features_dir, "*.pt"))
    g_track_ids = [os.path.splitext(os.path.basename(f))[0] for f in g_pt_files if os.path.exists(os.path.join(g_anno_dir, f"{os.path.splitext(os.path.basename(f))[0]}.jams"))]
    
    train_end_b = int(0.85 * len(valid_b_df))
    train_end_g = int(0.85 * len(g_track_ids))

    train_gold = ConcatDataset([
        BillboardDataset(valid_b_df.iloc[:train_end_b], cfg.data.billboard_features_dir, cfg.data.annotations_dir, augment=True, use_student_noise=True),
        GuitarSetDirectDataset(g_track_ids[:train_end_g], g_anno_dir, cfg.data.guitarset_features_dir, augment=True, use_student_noise=True)
    ])
    
    val_gold = ConcatDataset([
        BillboardDataset(valid_b_df.iloc[train_end_b:], cfg.data.billboard_features_dir, cfg.data.annotations_dir, augment=False),
        GuitarSetDirectDataset(g_track_ids[train_end_g:], g_anno_dir, cfg.data.guitarset_features_dir, augment=False)
    ])

    # --- 2. SILVER DATA (YouTube Pseudo-labels) ---
    if os.path.exists(cfg.data.pseudo_csv_path):
        train_silver = PseudoLabelDataset(cfg.data.pseudo_csv_path, cfg.data.youtube_features_dir, augment=True, use_student_noise=True)
    else:
        print(f"❌ Error: Pseudo-labels CSV not found at {cfg.data.pseudo_csv_path}")
        return

    # --- 3. ANCHOR SAMPLING (70% Gold / 30% Silver) ---
    train_full = ConcatDataset([train_gold, train_silver])
    weights = []
    w_gold = 0.70 / len(train_gold)
    w_silver = 0.30 / len(train_silver)
    for _ in range(len(train_gold)): weights.append(w_gold)
    for _ in range(len(train_silver)): weights.append(w_silver)
    
    sampler = WeightedRandomSampler(weights, num_samples=len(train_full), replacement=True)

    train_loader = DataLoader(train_full, batch_size=cfg.training.batch_size, sampler=sampler, collate_fn=pad_collate_fn, num_workers=4)
    val_loader = DataLoader(val_gold, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=4)

    # --- 4. INITIALIZE STUDENT (ENHANCED) ---
    model = EnhancedChordCRNN(num_classes=25, lstm_hidden=cfg.model.lstm_hidden, lstm_layers=cfg.model.lstm_layers, 
                              attention_heads=cfg.model.attention_heads, dropout=0.2).to(device)

    # Pre-load best Enhanced weights (to avoid restarting from scratch)
    _, teacher_file = get_historical_best_model(cfg.training.model_dir, args.teacher)
    if teacher_file:
        print(f"✅ Pre-loading weights from Enhanced Teacher: {teacher_file}")
        model.load_state_dict(torch.load(teacher_file, map_location=device, weights_only=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # --- 5. TRAINING ---
    trainer = ChordTrainer(model, optimizer, criterion, device, model_dir=cfg.training.model_dir, 
                           base_name=args.student_name, use_wandb=cfg.training.use_wandb)
    
    if cfg.training.use_wandb:
        import wandb
        wandb.init(project=cfg.training.project_name, name=f"hybrid-{args.student_name}", config=config_dict)

    trainer.fit(train_loader, val_loader, args.epochs)
    
    if trainer.best_file:
        model.load_state_dict(torch.load(trainer.best_file, map_location=device, weights_only=True))
        # FIX: Pass val_loader (DataLoader) instead of val_gold (Dataset)
        test_loss, test_acc = trainer.evaluate(val_loader) 
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("outputs.md", "a") as f:
            f.write(f"| {timestamp} | **{args.student_name.upper()}** | {test_loss:.4f} | {test_acc:.2f}% | (Hybrid Ensemble Cycle)\n")

    if cfg.training.use_wandb: wandb.finish()

if __name__ == "__main__":
    main()

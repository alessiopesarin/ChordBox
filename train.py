import torch
import torch.nn as nn
import pandas as pd
import os
import glob
import random
from datetime import datetime
from torch.utils.data import DataLoader

# Import architectures
from src.models.deep_crnn import DeepChordCRNN
from src.data.datasets import BillboardDataset, pad_collate_fn
from src.training.trainer import ChordTrainer
from src.utils.config_loader import load_config, Config

def main():
    # 1. Load Configuration
    config_dict = load_config("config/default.yaml")
    cfg = Config(config_dict)

    print("\n" + "="*40)
    print("🎸 TRAINING: MASTER RECOVERY (DEEP - GEN 0)")
    print("="*40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # --- DATA LOADING (Billboard Only) ---
    full_df = pd.read_csv(cfg.data.metadata_csv, dtype={'Track_ID': str})
    valid_rows = []
    for _, row in full_df.iterrows():
        tid = row['Track_ID']
        if os.path.exists(os.path.join(cfg.data.audio_dir, f"{tid}.wav")) and \
           os.path.exists(os.path.join(cfg.data.annotations_dir, f"{tid}.lab")):
            valid_rows.append(row)
            
    valid_df = pd.DataFrame(valid_rows)
    valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_end = int(0.85 * len(valid_df))
    val_end = int(0.95 * len(valid_df))
    
    train_ds = BillboardDataset(valid_df.iloc[:train_end], cfg.data.billboard_features_dir, cfg.data.annotations_dir, 
                                augment=True, chunk_frames=cfg.data.chunk_frames)
    val_ds = BillboardDataset(valid_df.iloc[train_end:val_end], cfg.data.billboard_features_dir, cfg.data.annotations_dir, 
                              augment=False, chunk_frames=cfg.data.chunk_frames)
    test_ds = BillboardDataset(valid_df.iloc[val_end:], cfg.data.billboard_features_dir, cfg.data.annotations_dir, 
                               augment=False, chunk_frames=cfg.data.chunk_frames)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=4)

    # --- INITIALIZE MODEL (DEEP) ---
    model = DeepChordCRNN(num_classes=25).to(device)

    # --- OPTIMIZER: ADAM (Back to basics) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # ignore_index=24: Standard logic for ACR models
    criterion = nn.CrossEntropyLoss(ignore_index=24)

    # --- TRAINING ---
    base_name = "master_deep_adam"
    trainer = ChordTrainer(model, optimizer, criterion, device, model_dir=cfg.training.model_dir, 
                           base_name=base_name, use_wandb=cfg.training.use_wandb)
    
    if cfg.training.use_wandb:
        import wandb
        wandb.init(project=cfg.training.project_name, name="master-deep-adam", config=config_dict)

    print(f"🚀 Training {base_name} for 40 epochs...")
    trainer.fit(train_loader, val_loader, 40)
    
    if trainer.best_file:
        model.load_state_dict(torch.load(trainer.best_file, map_location=device, weights_only=True))
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("outputs.md", "a") as f:
            f.write(f"| {timestamp} | **{base_name.upper()}** | {test_loss:.4f} | {test_acc:.2f}% | (Recovered Master - Adam)\n")

    if cfg.training.use_wandb: wandb.finish()

if __name__ == "__main__":
    main()

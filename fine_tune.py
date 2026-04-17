import torch
import torch.nn as nn
import pandas as pd
import os
import sys
import argparse
from datetime import datetime
from torch.utils.data import DataLoader

# Aggiunge la root al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.models.enhanced_crnn import EnhancedChordCRNN
from src.data.datasets import BillboardDataset, pad_collate_fn
from src.training.trainer import ChordTrainer
from src.utils.config_loader import load_config, Config

def get_historical_best_model(model_dir, base_name):
    import re
    pattern = rf"^{re.escape(base_name)}_([0-9]+\.[0-9]+)\.pth$"
    best_loss = float('inf')
    best_file = None
    for f in os.listdir(model_dir):
        match = re.match(pattern, f)
        if match:
            loss = float(match.group(1))
            if loss < best_loss:
                best_loss = loss
                best_file = os.path.join(model_dir, f)
    return best_loss, best_file

def main():
    # 1. Carica Config
    config_dict = load_config("config/default.yaml")
    cfg = Config(config_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*40)
    print("✨ CLEAN FINE-TUNING (POLISHING PHASE)")
    print("="*40)

    # 2. Carica Dati (Solo Billboard, NO NOISE)
    full_df = pd.read_csv(cfg.data.metadata_csv, dtype={'Track_ID': str})
    valid_rows = [row for _, row in full_df.iterrows() if os.path.exists(os.path.join(cfg.data.audio_dir, f"{row['Track_ID']}.wav"))]
    valid_df = pd.DataFrame(valid_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_end = int(0.85 * len(valid_df))
    val_end = int(0.95 * len(valid_df))
    
    # Notare: use_student_noise=False e augment=False (o molto leggero) per la massima precisione
    train_ds = BillboardDataset(valid_df.iloc[:train_end], cfg.data.billboard_features_dir, cfg.data.annotations_dir, 
                                augment=True, chunk_frames=cfg.data.chunk_frames, use_student_noise=False)
    val_ds = BillboardDataset(valid_df.iloc[train_end:val_end], cfg.data.billboard_features_dir, cfg.data.annotations_dir, 
                              augment=False, chunk_frames=cfg.data.chunk_frames, use_student_noise=False)
    test_ds = BillboardDataset(valid_df.iloc[val_end:], cfg.data.billboard_features_dir, cfg.data.annotations_dir, 
                               augment=False, chunk_frames=cfg.data.chunk_frames, use_student_noise=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate_fn)

    # 3. Modello
    model = EnhancedChordCRNN(num_classes=cfg.model.num_classes, lstm_hidden=cfg.model.lstm_hidden, 
                              lstm_layers=cfg.model.lstm_layers, attention_heads=cfg.model.attention_heads, dropout=0.2).to(device)

    # Carichiamo l'ultima Gen 4
    _, teacher_file = get_historical_best_model(cfg.training.model_dir, "student_enhanced_gen_4")
    if teacher_file:
        print(f"✅ Loading Gen 4 Weights for polishing: {teacher_file}")
        model.load_state_dict(torch.load(teacher_file, map_location=device, weights_only=True))
    else:
        print("❌ Error: Gen 4 model not found!")
        return

    # 4. Optimizer & Scheduler (Learning Rate Bassissimo)
    fine_tune_lr = 1e-5 
    optimizer = torch.optim.Adam(model.parameters(), lr=fine_tune_lr)
    criterion = nn.CrossEntropyLoss(ignore_index=24)

    # 5. Training
    trainer = ChordTrainer(model, optimizer, criterion, device, model_dir=cfg.training.model_dir, 
                           base_name="final_polished_model", use_wandb=cfg.training.use_wandb)
    
    if cfg.training.use_wandb:
        import wandb
        wandb.init(project=cfg.training.project_name, name="final-polishing", config=config_dict)

    print(f"🚀 Starting polishing for 10 epochs with LR: {fine_tune_lr}")
    trainer.fit(train_loader, val_loader, 10)
    
    # 6. Eval
    if trainer.best_file:
        model.load_state_dict(torch.load(trainer.best_file, map_location=device, weights_only=True))
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("outputs.md", "a") as f:
            f.write(f"| {timestamp} | **POLISHED (GEN 4)** | {test_loss:.4f} | {test_acc:.2f}% | (Final Polishing Phase)\n")
    
    if cfg.training.use_wandb: wandb.finish()

if __name__ == "__main__":
    main()

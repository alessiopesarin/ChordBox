import torch
import torch.nn as nn
import pandas as pd
import os
import glob
import random
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset

# Import all available architectures
from src.models.enhanced_crnn import EnhancedChordCRNN
from src.data.datasets import BillboardDataset, PseudoLabelDataset, pad_collate_fn
from src.training.trainer import ChordTrainer
from src.utils.config_loader import load_config, Config

def get_historical_best_model(model_dir, base_name):
    """Finds the best model file dynamically."""
    import re
    pattern = rf"^{re.escape(base_name)}_([0-9]+\.[0-9]+)\.pth$"
    best_loss = float('inf')
    best_file = None
    
    if not os.path.exists(model_dir):
        return None, None

    for f in os.listdir(model_dir):
        match = re.match(pattern, f)
        if match:
            loss = float(match.group(1))
            if loss < best_loss:
                best_loss = loss
                best_file = os.path.join(model_dir, f)
    return best_loss, best_file

def main():
    parser = argparse.ArgumentParser(description="Train a Student model using pseudo-labels.")
    parser.add_argument("--teacher", type=str, default="enhanced", help="Base name of the teacher model for weight initialization")
    parser.add_argument("--student_name", type=str, default="student_enhanced_gen_2", help="Base name for the new student model")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    args = parser.parse_args()

    # 1. Load Configuration
    config_dict = load_config("config/default.yaml")
    cfg = Config(config_dict)

    print("\n" + "="*40)
    print(f"🎓 NOISY STUDENT: {args.student_name.upper()}")
    print("="*40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # --- DATA LOADING & PREPARATION ---
    # 1. Billboard (Teacher data - cleaned)
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
    
    train_billboard = BillboardDataset(
        valid_df.iloc[:train_end], 
        cfg.data.billboard_features_dir, 
        cfg.data.annotations_dir, 
        augment=True,
        chunk_frames=cfg.data.chunk_frames,
        use_student_noise=True
    )
    val_billboard = BillboardDataset(
        valid_df.iloc[train_end:val_end], 
        cfg.data.billboard_features_dir, 
        cfg.data.annotations_dir, 
        augment=False,
        chunk_frames=cfg.data.chunk_frames
    )
    test_billboard = BillboardDataset(
        valid_df.iloc[val_end:], 
        cfg.data.billboard_features_dir, 
        cfg.data.annotations_dir, 
        augment=False,
        chunk_frames=cfg.data.chunk_frames
    )

    # 2. YouTube (Pseudo-labeled data)
    if os.path.exists(cfg.data.pseudo_csv_path):
        print(f"✅ Loading pseudo-labels from {cfg.data.pseudo_csv_path}")
        youtube_dataset = PseudoLabelDataset(
            cfg.data.pseudo_csv_path,
            cfg.data.youtube_features_dir,
            chunk_frames=cfg.data.chunk_frames,
            augment=True,
            use_student_noise=True
        )
    else:
        print(f"❌ Error: Pseudo-labels CSV not found at {cfg.data.pseudo_csv_path}")
        return

    # 3. Merge and Dataloaders
    train_dataset = ConcatDataset([train_billboard, youtube_dataset])
    
    # --- BALANCED SAMPLING ---
    # Billboard is the 'anchor' (high quality), YouTube is the 'volume' (noisy).
    # We want each batch to have ~50% from each to prevent YouTube from overwhelming Billboard.
    weights = []
    # Weight for Billboard samples
    w_billboard = 1.0 / len(train_billboard)
    # Weight for YouTube samples
    w_youtube = 1.0 / len(youtube_dataset)
    
    for i in range(len(train_billboard)):
        weights.append(w_billboard)
    for i in range(len(youtube_dataset)):
        weights.append(w_youtube)
        
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(train_dataset), replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        sampler=sampler, # Using sampler instead of shuffle=True
        collate_fn=pad_collate_fn, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_billboard, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        collate_fn=pad_collate_fn, 
        num_workers=4
    )
    test_loader = DataLoader(
        test_billboard, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        collate_fn=pad_collate_fn, 
        num_workers=4
    )

    print(f"📊 DATASET SUMMARY:")
    print(f"   - Labeled (Billboard): {len(train_billboard)} tracks")
    print(f"   - Pseudo-labeled (YouTube): {len(youtube_dataset)} tracks")
    print(f"   - Total Training: {len(train_dataset)} tracks")

    # --- INITIALIZE STUDENT MODEL (ENHANCED) ---
    # Lowered dropout to 0.2 to help stabilize learning after Noisy Student cycles
    model = EnhancedChordCRNN(
        num_classes=cfg.model.num_classes,
        lstm_hidden=cfg.model.lstm_hidden,
        lstm_layers=cfg.model.lstm_layers,
        attention_heads=cfg.model.attention_heads,
        dropout=0.2 
    ).to(device)

    # Load Teacher Weights
    _, teacher_file = get_historical_best_model(cfg.training.model_dir, args.teacher)
    if teacher_file:
        print(f"✅ Pre-loading weights from Teacher: {teacher_file}")
        model.load_state_dict(torch.load(teacher_file, map_location=device, weights_only=True))
    else:
        print(f"⚠️ No teacher model found with name '{args.teacher}'. Starting from scratch.")

    # Aggressive Learning Rate logic
    if "gen_2" in args.student_name:
        lr_factor = 2.5
    elif "gen_3" in args.student_name:
        lr_factor = 1.5
    else:
        lr_factor = 1.0
        
    final_lr = cfg.training.learning_rate * lr_factor
    print(f"🚀 Using learning rate: {final_lr}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=final_lr)
    
    # --- CRITICAL: Changed ignore_index to -100 ---
    # This allows the model to learn N.C. (24) from Billboard data, 
    # while ignoring YouTube frames marked as -100 by the Dataloader.
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # --- TRAINING ---
    trainer = ChordTrainer(
        model, optimizer, criterion, device, 
        model_dir=cfg.training.model_dir, 
        base_name=args.student_name,
        use_wandb=cfg.training.use_wandb
    )
    
    if cfg.training.use_wandb:
        import wandb
        wandb.init(project=cfg.training.project_name, name=f"noisy-{args.student_name}", config=config_dict)

    trainer.fit(train_loader, val_loader, args.epochs)
    
    # --- FINAL EVALUATION ---
    if trainer.best_file:
        print(f"🏆 Best Student model found! Running final evaluation...")
        model.load_state_dict(torch.load(trainer.best_file, map_location=device, weights_only=True))
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        # --- RECORD OUTPUTS ---
        outputs_file = "outputs.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(outputs_file, "a") as f:
            f.write(f"| {timestamp} | **{args.student_name.upper()}** | {test_loss:.4f} | {test_acc:.2f}% | (Noisy Student Cycle)\n")
        
        print(f"📝 Results appended to {outputs_file}")

    if cfg.training.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()

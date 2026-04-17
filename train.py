import torch
import torch.nn as nn
import pandas as pd
import os
import glob
import random
from datetime import datetime
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

# Import architectures and datasets
from src.models.deep_crnn import DeepChordCRNN
from src.data.datasets import BillboardDataset, GuitarSetDirectDataset, pad_collate_fn
from src.training.trainer import ChordTrainer
from src.utils.config_loader import load_config, Config

def main():
    # 1. Load Configuration
    config_dict = load_config("config/default.yaml")
    cfg = Config(config_dict)

    print("\n" + "="*40)
    print("🎸 TRAINING: SUPER MASTER (DEEP - GEN 0)")
    print("      (Billboard + GuitarSet Balanced)")
    print("="*40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # --- 1. BILLBOARD DATA ---
    full_df = pd.read_csv(cfg.data.metadata_csv, dtype={'Track_ID': str})
    valid_billboard = []
    for _, row in full_df.iterrows():
        tid = row['Track_ID']
        if os.path.exists(os.path.join(cfg.data.audio_dir, f"{tid}.wav")) and \
           os.path.exists(os.path.join(cfg.data.annotations_dir, f"{tid}.lab")):
            valid_billboard.append(row)
            
    billboard_df = pd.DataFrame(valid_billboard).sample(frac=1, random_state=42).reset_index(drop=True)
    b_train_end = int(0.85 * len(billboard_df))
    b_val_end = int(0.95 * len(billboard_df))

    train_billboard = BillboardDataset(billboard_df.iloc[:b_train_end], cfg.data.billboard_features_dir, cfg.data.annotations_dir, 
                                       augment=True, chunk_frames=cfg.data.chunk_frames)
    val_billboard = BillboardDataset(billboard_df.iloc[b_train_end:b_val_end], cfg.data.billboard_features_dir, cfg.data.annotations_dir, 
                                     augment=False, chunk_frames=cfg.data.chunk_frames)
    test_billboard = BillboardDataset(billboard_df.iloc[b_val_end:], cfg.data.billboard_features_dir, cfg.data.annotations_dir, 
                                      augment=False, chunk_frames=cfg.data.chunk_frames)

    # --- 2. GUITARSET DATA ---
    g_features_dir = cfg.data.guitarset_features_dir
    g_anno_dir = cfg.data.guitarset_annotations_dir
    g_pt_files = glob.glob(os.path.join(g_features_dir, "*.pt"))
    g_track_ids = [os.path.splitext(os.path.basename(f))[0] for f in g_pt_files]
    g_track_ids = [tid for tid in g_track_ids if os.path.exists(os.path.join(g_anno_dir, f"{tid}.jams"))]
    
    random.seed(42)
    random.shuffle(g_track_ids)
    g_train_end = int(0.85 * len(g_track_ids))
    g_val_end = int(0.95 * len(g_track_ids))
    
    train_guitarset = GuitarSetDirectDataset(g_track_ids[:g_train_end], g_anno_dir, g_features_dir, augment=True, chunk_frames=cfg.data.chunk_frames)
    val_guitarset = GuitarSetDirectDataset(g_track_ids[g_train_end:g_val_end], g_anno_dir, g_features_dir, augment=False, chunk_frames=cfg.data.chunk_frames)
    test_guitarset = GuitarSetDirectDataset(g_track_ids[g_val_end:], g_anno_dir, g_features_dir, augment=False, chunk_frames=cfg.data.chunk_frames)

    # --- 3. MERGE WITH BALANCED SAMPLING ---
    train_dataset = ConcatDataset([train_billboard, train_guitarset])
    
    # Calculate weights to have 50% Billboard and 50% GuitarSet in batches
    weights = []
    w_b = 1.0 / len(train_billboard)
    w_g = 1.0 / len(train_guitarset)
    for _ in range(len(train_billboard)): weights.append(w_b)
    for _ in range(len(train_guitarset)): weights.append(w_g)
    
    sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, sampler=sampler, collate_fn=pad_collate_fn, num_workers=4)
    
    # Validation and Test
    val_dataset = ConcatDataset([val_billboard, val_guitarset])
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=4)
    
    # We keep test loaders separate to see individual performance
    test_loader_b = DataLoader(test_billboard, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=4)
    test_loader_g = DataLoader(test_guitarset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=4)

    print(f"📊 DATASET SUMMARY:")
    print(f"   - Billboard (Gold-Mix): {len(train_billboard)} train samples")
    print(f"   - GuitarSet (Gold-Solo): {len(train_guitarset)} train samples")
    print(f"   - Total Training (Balanced): {len(train_dataset)}")

    # --- INITIALIZE MODEL ---
    model = DeepChordCRNN(num_classes=25).to(device)

    # --- OPTIMIZER ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=24)

    # --- TRAINING ---
    base_name = "super_master_deep_balanced"
    trainer = ChordTrainer(model, optimizer, criterion, device, model_dir=cfg.training.model_dir, 
                           base_name=base_name, use_wandb=cfg.training.use_wandb)
    
    if cfg.training.use_wandb:
        import wandb
        wandb.init(project=cfg.training.project_name, name=base_name, config=config_dict)

    print(f"🚀 Training {base_name} for 60 epochs...")
    trainer.fit(train_loader, val_loader, 60)
    
    if trainer.best_file:
        model.load_state_dict(torch.load(trainer.best_file, map_location=device, weights_only=True))
        
        print("\n--- Final Evaluation on Billboard ---")
        loss_b, acc_b = trainer.evaluate(test_loader_b)
        
        print("\n--- Final Evaluation on GuitarSet ---")
        loss_g, acc_g = trainer.evaluate(test_loader_g)
        
        avg_acc = (acc_b + acc_g) / 2
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("outputs.md", "a") as f:
            f.write(f"| {timestamp} | **{base_name.upper()}** | B:{loss_b:.2f}/G:{loss_g:.2f} | B:{acc_b:.2f}%/G:{acc_g:.2f}% | (Super Master Balanced)\n")

    if cfg.training.use_wandb: wandb.finish()

if __name__ == "__main__":
    main()

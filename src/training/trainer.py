import torch
import torch.nn as nn
import os
import glob
import re
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class ChordTrainer:
    def __init__(self, model, optimizer, criterion, device, model_dir="./models", base_name="baseline", use_wandb=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_dir = model_dir
        self.base_name = base_name
        self.best_val_loss = float('inf')
        self.best_file = None
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        os.makedirs(self.model_dir, exist_ok=True)
        self._load_historical_best()

    def _load_historical_best(self):
        # Reset local records
        self.best_val_loss = float('inf')
        self.best_file = None
        
        # Strictly look for files: {base_name}_{loss}.pth
        if not os.path.exists(self.model_dir):
            return

        files = [f for f in os.listdir(self.model_dir) if f.endswith(".pth")]
        
        for f in files:
            # Match: base_name_NUMBER.pth
            # ^ starts with base_name, followed by _, followed by float, followed by .pth $
            pattern = rf"^{re.escape(self.base_name)}_([0-9]+\.[0-9]+)\.pth$"
            match = re.match(pattern, f)
            if match:
                loss = float(match.group(1))
                if loss < self.best_val_loss:
                    self.best_val_loss = loss
                    self.best_file = os.path.join(self.model_dir, f)
        
        if self.best_file:
            print(f"✅ Found existing record for {self.base_name}: {self.best_val_loss:.4f} ({os.path.basename(self.best_file)})")
        else:
            print(f"🆕 No previous record found for architecture: {self.base_name}")

    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0.0
        valid_batches = 0
        
        for chromas, labels in train_loader:
            chromas, labels = chromas.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(chromas)
            num_classes = outputs.shape[-1]
            loss = self.criterion(outputs.view(-1, num_classes), labels.view(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            train_loss += loss.item()
            valid_batches += 1
            
        return (train_loss / valid_batches) if valid_batches > 0 else float('inf')

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        valid_batches = 0
        with torch.no_grad():
            for chromas, labels in val_loader:
                chromas, labels = chromas.to(self.device), labels.to(self.device)
                outputs = self.model(chromas)
                num_classes = outputs.shape[-1]
                loss = self.criterion(outputs.view(-1, num_classes), labels.view(-1))
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                    
                val_loss += loss.item()
                valid_batches += 1
        return (val_loss / valid_batches) if valid_batches > 0 else float('inf')

    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        valid_batches = 0
        correct = 0
        total = 0
        
        print("\n" + "="*40)
        print("🚀 RUNNING EVALUATION...")
        
        with torch.no_grad():
            for chromas, labels in test_loader:
                chromas, labels = chromas.to(self.device), labels.to(self.device)
                outputs = self.model(chromas)
                num_classes = outputs.shape[-1]
                
                loss = self.criterion(outputs.view(-1, num_classes), labels.view(-1))
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Warning: NaN/Inf loss detected in evaluation batch. Skipping...")
                    continue
                
                test_loss += loss.item()
                valid_batches += 1
                
                preds = torch.argmax(outputs, dim=-1)
                mask = labels != 24 
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
                
        avg_loss = (test_loss / valid_batches) if valid_batches > 0 else float('nan')
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        
        print(f"✅ Loss: {avg_loss:.4f}")
        print(f"🎯 Accuracy (Ignoring N.C.): {accuracy:.2f}%")
        print("="*40 + "\n")
        
        if self.use_wandb:
            wandb.log({"test_loss": avg_loss, "test_accuracy": accuracy})
            
        return avg_loss, accuracy

    def save_checkpoint(self, val_loss):
        if val_loss < self.best_val_loss:
            print(f"  -> OVERALL RECORD BROKEN! (Val Loss dropped to {val_loss:.4f}). Saving new model...")
            
            old_files = glob.glob(os.path.join(self.model_dir, f"{self.base_name}_*.pth"))
            for old_f in old_files:
                try: os.remove(old_f)
                except: pass
            
            new_filename = f"{self.base_name}_{val_loss:.4f}.pth"
            self.best_file = os.path.join(self.model_dir, new_filename)
            torch.save(self.model.state_dict(), self.best_file)
            self.best_val_loss = val_loss
        else:
            diff = val_loss - self.best_val_loss
            print(f"  -> Model did not beat the historical best (+{diff:.4f}).")

    def fit(self, train_loader, val_loader, epochs):
        # Switch to CosineAnnealingLR: decays following a cosine curve down to eta_min
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs, 
            eta_min=1e-6
        )
        
        for epoch in range(epochs):
            avg_train_loss = self.train_epoch(train_loader)
            avg_val_loss = self.validate(val_loader)
            
            # Scheduler step is now based on epoch, not val_loss
            scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
            
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": current_lr
                })
                
            self.save_checkpoint(avg_val_loss)
            
        print(f"\nTraining complete. The best model is at: {self.best_file}")
        return self.best_val_loss

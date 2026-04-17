# 🎸 Deep Chord Training Guide

Questa guida spiega come configurare e avviare il training scegliendo tra le diverse architetture disponibili nel progetto.

## 🛠️ 1. Configurazione del Modello
Tutte le impostazioni di training si trovano in `config/default.yaml`. Per cambiare architettura, modifica il campo `base_model_name` sotto la sezione `training`.

```yaml
training:
  base_model_name: "enhanced"  # Opzioni: "baseline", "deep", "enhanced"
  epochs: 40
  batch_size: 16
  learning_rate: 0.0001
```

### Architetture Disponibili:
1.  **`baseline`** (`ChordCRNN`): 
    * CNN semplice (2 layer) + LSTM (2 layer).
    * Leggero e veloce, ideale per test rapidi o per macchine con poca GPU.
2.  **`deep`** (`DeepChordCRNN`):
    * CNN profonda (4 layer) + LSTM (3 layer).
    * Ottimo bilanciamento tra complessità e performance. Corrisponde ai modelli storici da 19MB.
3.  **`enhanced`** (`EnhancedChordCRNN`):
    * **Architettura SOTA attuale**.
    * Residual CNN + Self-Attention + LayerNorm.
    * Richiede più memoria GPU ma offre la migliore precisione e stabilità temporale.

## 🚀 2. Avviare il Training
Una volta configurato il file `.yaml`, avvia semplicemente lo script principale:

```bash
python train.py
```

Il sistema caricherà automaticamente:
* I dati grezzi da `data/raw/`
* Le feature precompute da `data/processed/`
* L'architettura scelta nel config.

## 📈 3. Monitoraggio (WandB)
Se hai impostato `use_wandb: true`, puoi monitorare le metriche (loss, accuracy) in tempo reale sulla tua dashboard di Weights & Biases.

## 📁 4. Risultati e Checkpoint
I modelli salvati verranno creati nella cartella `models/` con un nome che include la loss di validazione (es. `enhanced_1.4205.pth`). Il `ChordTrainer` tiene traccia automaticamente del record storico per evitare di sovrascrivere modelli migliori.

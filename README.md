# 📦 ChordBox

ChordBox is a State-of-the-Art (SOTA) system for **Automatic Chord Recognition** using Deep Learning. It leverages a Hybrid CRNN architecture (CNN + Bi-LSTM + Self-Attention) and an iterative **Noisy Student** training pipeline to achieve high precision and robustness on diverse musical audio.

---

## 🚀 Key Features

*   **Advanced Hybrid Architecture**: Combines ResNet-style CNN for local feature extraction, Bi-LSTM for temporal dependencies, and Self-Attention for global harmonic context.
*   **Noisy Student Pipeline**: Automated iterative training cycle using 1500+ unlabeled YouTube tracks to expand the knowledge base beyond small human-annotated datasets.
*   **Ensemble Labeling**: Multi-model consensus system (Deep + Enhanced) to generate high-confidence pseudo-labels.
*   **Balanced Training**: Weighted sampling strategy to maintain an anchor on "Gold Standard" datasets (Billboard/GuitarSet) while learning from large-scale unlabeled data.
*   **Full Audio Pipeline**: From raw YouTube download and CQT pre-computation to real-time-ready inference with logit-averaging for flicker-free predictions.

---

## 🛠️ Project Structure

```text
├── src/
│   ├── models/       # CRNN, Deep, and Enhanced architectures
│   ├── training/     # Trainer, Pseudo-label generator, Orchestrator
│   ├── data/         # PyTorch Datasets and Dataloaders
│   ├── inference/    # Predictor and Smoothing logic
│   └── utils/        # Audio processing and Config loaders
├── config/           # YAML configuration files
├── util/             # Data scrapers and pre-computation scripts
├── models/           # Saved checkpoints (.pth)
├── data/
│   ├── raw/          # Original audio and annotations
│   └── processed/    # Pre-computed CQT tensors (.pt)
├── train.py          # Master model training script
├── train_student.py  # Student self-training script
└── orchestrator.py   # Automated Noisy Student loop
```

---

## 📦 Installation & Setup

1.  **Environment**:
    ```bash
    conda create -n deep-chord python=3.10
    conda activate deep-chord
    pip install torch librosa pandas tqdm wandb yt-dlp scipy pyyaml
    ```

2.  **Configuration**:
    Edit `config/default.yaml` to set your paths, model hyperparameters, and WandB preferences.

---

## 🏋️ Training Workflow

### 1. The Honest Master (Gen 0)
Train the initial teacher model on high-quality labeled data (Billboard):
```bash
python3 train.py
```

### 2. Operation Massive YouTube
Download unlabeled audio and pre-compute features:
```bash
python3 util/download_1001.py
python3 src/utils/precompute_youtube.py
```

### 3. Automated Self-Training Cycle
Launch the orchestrator to generate pseudo-labels and evolve the student:
```bash
python3 orchestrator.py --start_gen 1 --num_loops 3 --teacher_enh honest_master
```

---

## 🎯 Inference

Use the `ChordPredictor` for stable, flicker-free predictions on any audio file:

```python
from src.inference.predictor import ChordPredictor

predictor = ChordPredictor(model_path="models/your_best_model.pth")
chords = predictor.predict_audio("path/to/song.wav")
print(chords)
```

---

## 📜 Evolution Log
For detailed technical notes on the model's evolution, bug fixes, and architectural experiments, refer to [EVOLUTION_LOG.md](./EVOLUTION_LOG.md).

## 🤖 AI Context
If you are using this project with an AI Assistant, please refer to [AI_CONTEXT.md](./AI_CONTEXT.md) for foundational mandates and architectural conventions.

# 🚀 Project Todo: Breaking the 60% Barrier

## 🎯 Objective
Restore the record performance of **57.95%** and surpass **60%** by allowing the Attention mechanism to learn without being limited by the Deep Master model.

## 🛠️ Phase 1: Attention Liberation (Gen 9)
- [ ] **Update `generate_pseudo_labels.py`**:
    - Remove the Ensemble logic (or make it optional).
    - Use exclusively the **Student Gen 6** model (current record holder at 57.95%) as the Teacher.
    - Set confidence threshold to **0.72** to capture more complex harmonic transitions.
- [ ] **Configure `orchestrator.py`**:
    - Point `--teacher_enh` to `student_enhanced_gen_6`.
    - Ensure it uses the solo-labeling mode.
- [ ] **Execution**:
    - Run labeling on the full YouTube dataset (~15-20 min with current optimizations).
    - Train **Student Gen 9** using the 70/30 balanced ratio.

## 🧪 Phase 2: Final Polishing
- [ ] **Gentle Fine-tuning**:
    - If Gen 9 reaches ~59%, run `fine_tune.py` on Gold data ONLY (Billboard + GuitarSet).
    - Use an ultra-low learning rate (5e-6) for 10 epochs to "lock in" the precision.

## 📝 Technical Notes
- **Ignore Index**: Keep using `-100` to allow the model to learn silence (24) from Billboard only.
- **Normalization**: Ensure `(x - mean) / std` consistency is maintained across all datasets.
- **Balanced Sampling**: Keep the 70/30 ratio to prevent noisy labels from overwhelming the human-annotated data.

---
*Refer to [EVOLUTION_LOG.md](./EVOLUTION_LOG.md) for the full history of experiments.*

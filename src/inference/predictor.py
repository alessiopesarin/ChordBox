import torch
import librosa
import numpy as np
import os
import glob
import re
from scipy.stats import mode

from src.models.crnn import ChordCRNN
from src.models.deep_crnn import DeepChordCRNN
from src.models.enhanced_crnn import EnhancedChordCRNN
from src.models.multitask_crnn import MultiTaskChordCRNN
from src.data.datasets import CHORDS, INT_TO_CHORD
from src.utils.audio import audio_to_tensor

# Multi-task dictionaries (needed for root/quality decoding)
ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'N.C.']
QUALITIES = ['maj', 'min', 'maj7', '7', 'min7', 'N.C.']
INT_TO_ROOT = {i: r for i, r in enumerate(ROOTS)}
INT_TO_QUAL = {i: q for i, q in enumerate(QUALITIES)}

class ChordPredictor:
    def __init__(self, model_path=None, device=None, config=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.model = None
        self.model_type = "standard" # or "multitask"
        
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Loads a model and detects its architecture from the path or filename."""
        print(f"Loading model from: {model_path}")
        
        # Heuristic to detect architecture
        model_path_lower = model_path.lower()
        num_classes = self.config.model.num_classes if self.config else 25
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)

        # 1. Try Heuristic first
        if "multitask" in model_path_lower:
            self.model = MultiTaskChordCRNN(num_roots=13, num_quals=6).to(self.device)
            self.model_type = "multitask"
        elif "baseline" in model_path_lower or "transfer" in model_path_lower:
            self.model = ChordCRNN(num_classes=num_classes).to(self.device)
            self.model_type = "baseline"
        elif "best" in model_path_lower or "student" in model_path_lower:
            # Many 'best' and 'student' models use the deep backbone but single task
            self.model = DeepChordCRNN(num_classes=num_classes).to(self.device)
            self.model_type = "deep"
        else:
            self.model = EnhancedChordCRNN(num_classes=num_classes).to(self.device)
            self.model_type = "enhanced"

        # 2. Try loading, and fallback if it fails
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            print(f"⚠️ Heuristic '{self.model_type}' failed for {model_path}. Trying fallback architectures...")
            
            # Try all architectures in sequence
            architectures = [
                ("enhanced", lambda: EnhancedChordCRNN(num_classes=num_classes)),
                ("deep", lambda: DeepChordCRNN(num_classes=num_classes)),
                ("baseline", lambda: ChordCRNN(num_classes=num_classes)),
                ("multitask", lambda: MultiTaskChordCRNN(num_roots=13, num_quals=6))
            ]
            
            success = False
            for arch_name, arch_factory in architectures:
                if arch_name == self.model_type: continue # Skip what we already tried
                
                try:
                    temp_model = arch_factory().to(self.device)
                    temp_model.load_state_dict(state_dict)
                    self.model = temp_model
                    self.model_type = arch_name
                    print(f"✅ Successfully loaded as {arch_name} architecture.")
                    success = True
                    break
                except Exception:
                    continue
            
            if not success:
                # If everything failed, re-try original to show the error
                print(f"❌ Failed to load model {model_path} with any known architecture.")
                self.model = EnhancedChordCRNN(num_classes=num_classes).to(self.device)
                self.model.load_state_dict(state_dict) # This will raise the original RuntimeError

        self.model.eval()
        print(f"Model loaded successfully ({self.model_type} architecture).")

    def _normalize_chroma(self, chroma):
        """Mandatory normalization logic as per README lessons learned."""
        return (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-8)

    def predict_audio(self, audio_path, chunk_frames=430):
        """Full pipeline: Audio -> CQT -> Chunks -> Inference -> Smoothing."""
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512).T
        chroma = self._normalize_chroma(chroma)
        
        return self.predict_chroma(chroma, chunk_frames)

    def predict_chroma(self, chroma, chunk_frames=430, overlap_frames=150):
        """Inference with Sliding Window and Logit Averaging for maximum stability."""
        if self.model is None:
            raise Exception("No model loaded in ChordPredictor!")

        time_frames = chroma.shape[0]
        num_classes = 25 if self.model_type != "multitask" else 13 * 100 # simplified for logic
        
        # Buffer to accumulate logits and counter for averaging
        logit_accumulator = np.zeros((time_frames, 25)) 
        count_accumulator = np.zeros(time_frames)
        
        step = chunk_frames - overlap_frames
        
        for i in range(0, time_frames, step):
            end_idx = min(i + chunk_frames, time_frames)
            chunk = chroma[i:end_idx, :]
            
            actual_len = chunk.shape[0]
            if actual_len < 5: continue # Too short
            
            # Padding
            if actual_len < chunk_frames:
                pad = chunk_frames - actual_len
                chunk = np.pad(chunk, ((0, pad), (0, 0)), mode='edge')
            
            chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.model_type == "multitask":
                    # For multitask we take only hard predictions
                    # (averaging logits across two different tasks is complex to handle here)
                    out_r, out_q = self.model(chunk_tensor)
                    r_preds = torch.argmax(out_r, dim=-1).squeeze(0).cpu().numpy()[:actual_len]
                    q_preds = torch.argmax(out_q, dim=-1).squeeze(0).cpu().numpy()[:actual_len]
                    
                    # Heuristic mapping to fill logit_accumulator (dummy one-hot)
                    for t_idx in range(actual_len):
                        val = r_preds[t_idx] # simplified: use only root for multitask in this accumulator
                        logit_accumulator[i + t_idx, val] += 1.0
                        count_accumulator[i + t_idx] += 1
                else:
                    logits = self.model(chunk_tensor) # [1, T, 25]
                    logits = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()[:actual_len]
                    
                    logit_accumulator[i : i + actual_len, :] += logits
                    count_accumulator[i : i + actual_len] += 1

        # Calculate average probabilities
        final_probs = logit_accumulator / np.expand_dims(count_accumulator + 1e-8, axis=-1)
        final_preds = np.argmax(final_probs, axis=-1)
        
        # If logits were all zero (e.g., multitask not handled well), argmax will give 0.
        # Safety correction:
        final_preds[count_accumulator == 0] = 24 # N.C.

        # Post-processing: Increased kernel size to eliminate flickering
        # Use kernel 15 (approx 0.3s) which is a good compromise
        smoothed = self._smooth_predictions(final_preds, kernel_size=15)
        
        return self._decode_predictions(smoothed)

    def _smooth_predictions(self, preds, kernel_size=15): # Increased to 15, as in the old script
        """Statistical filter: eliminates noise using the Mode (majority)."""
        pad_size = kernel_size // 2
        padded = np.pad(preds, (pad_size, pad_size), mode='edge')
        smoothed = np.zeros_like(preds)
        
        for i in range(len(preds)):
            window = padded[i : i + kernel_size]
            m = mode(window)[0]
            smoothed[i] = m[0] if isinstance(m, np.ndarray) else m
        return smoothed

    def _decode_predictions(self, preds):
        """Converts integer indices back to chord names."""
        chord_names = []
        if self.model_type == "multitask":
            for p in preds:
                r_idx, q_idx = p // 100, p % 100
                r_name = INT_TO_ROOT[r_idx]
                q_name = INT_TO_QUAL[q_idx]
                if r_name == 'N.C.':
                    chord_names.append('N.C.')
                else:
                    suffix = {"maj": "", "min": "m", "maj7": "maj7", "7": "7", "min7": "m7"}.get(q_name, "")
                    chord_names.append(r_name + suffix)
        else:
            chord_names = [INT_TO_CHORD[i] for i in preds]
        return chord_names

    @staticmethod
    def format_to_regions(chord_names, frame_duration=512/22050.0):
        """
        No temporal filtering or "Sandwich" logic. We rely 
        on _smooth_predictions for stability.
        """
        if not chord_names: return []
        
        regions = []
        current_chord = chord_names[0]
        start_time = 0.0
        
        for i, chord in enumerate(chord_names):
            if chord != current_chord:
                end_time = i * frame_duration
                regions.append({"start": start_time, "end": end_time, "chord": current_chord})
                current_chord = chord
                start_time = end_time
                
        # Close the last region
        regions.append({"start": start_time, "end": len(chord_names) * frame_duration, "chord": current_chord})
        
        return regions

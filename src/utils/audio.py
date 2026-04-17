import librosa
import torch

def audio_to_tensor(audio_path, sr=22050, hop_length=512):
    """
    Standard function to convert audio file to CQT tensor.
    """
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T
    # Note: Normalization is handled during pre-computation or inside the model if needed,
    # but here we return the raw CQT for inference.
    chroma_tensor = torch.tensor(chroma, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return chroma_tensor, chroma.shape[0]

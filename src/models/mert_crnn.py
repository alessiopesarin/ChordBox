import torch
import torch.nn as nn

class MERTChordCRNN(nn.Module):
    """
    Architecture optimized for pre-extracted features from MERT (Music2Vec).
    Input shape expected: [Batch, Time, 768]
    """
    def __init__(self, num_classes=25, input_dim=768, lstm_hidden=256, lstm_layers=2, attention_heads=4, dropout=0.4):
        super(MERTChordCRNN, self).__init__()
        
        # --- 1. INITIAL PROJECTION ---
        self.projection = nn.Sequential(
            nn.Linear(input_dim, lstm_hidden * 2),
            nn.LayerNorm(lstm_hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # --- 2. TEMPORAL BLOCK (LSTM) ---
        self.rnn = nn.LSTM(
            input_size=lstm_hidden * 2, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # --- 3. SELF-ATTENTION BLOCK (Pre-Norm) ---
        self.layer_norm_attn = nn.LayerNorm(lstm_hidden * 2)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2, 
            num_heads=attention_heads, 
            batch_first=True
        )
        
        # --- 4. CLASSIFIER ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128), 
            nn.LayerNorm(128),   
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input x might be [Batch, 1, Time, 768] due to pad_collate_fn
        if x.dim() == 4:
            x = x.squeeze(1)
            
        x = self.projection(x)
        
        # Temporal dependencies
        x, _ = self.rnn(x)
        
        # Self-Attention with Residual & Pre-Norm
        x_norm = self.layer_norm_attn(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Final classification
        x = self.classifier(x)
        return x

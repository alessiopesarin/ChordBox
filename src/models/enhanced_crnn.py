import torch
import torch.nn as nn

class EnhancedChordCRNN(nn.Module):
    """
    Enhanced Architecture: Deep CNN with Residual Connections + Bi-LSTM + Self-Attention.
    """
    def __init__(self, num_classes=25, lstm_hidden=256, lstm_layers=2, attention_heads=4, dropout=0.3):
        super(EnhancedChordCRNN, self).__init__()
        
        # --- 1. RESIDUAL CNN BLOCK ---
        self.conv_init = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Residual Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # Transition to 128 channels
        self.conv_transition = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128)
        )
        
        # Residual Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.relu = nn.ReLU()
        
        # --- 2. TEMPORAL BLOCK (LSTM) ---
        self.rnn = nn.LSTM(
            input_size=128 * 3, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0      
        )
        
        # --- 3. SELF-ATTENTION MECHANISM ---
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2, 
            num_heads=attention_heads, 
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)
        
        # --- 4. CLASSIFIER ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128), 
            nn.LayerNorm(128),   
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 1. Convolutional Layer with Residuals
        x = self.conv_init(x)
        
        res = x
        x = self.conv1(x)
        x = self.relu(x + res) 
        x = self.pool1(x)
        
        x = self.conv_transition(x)
        
        res = x
        x = self.conv2(x)
        x = self.relu(x + res) 
        x = self.pool2(x)
        
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, -1)
        
        # 2. LSTM
        x, _ = self.rnn(x)
        
        # 3. Self-Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.layer_norm(x + attn_out)
        
        # 4. Classifier
        x = self.classifier(x)
        return x

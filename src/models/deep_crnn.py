import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Convolutional block with residual connection (skip connection)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If we change the number of channels, we must align the skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Residual connection
        out = self.relu(out)
        return out

class DeepChordCRNN(nn.Module):
    """
    Ultra-Robust Deep Architecture (No Attention): 
    ResNet CNN + Spatial Dropout + 3-Layer Bi-LSTM + LayerNorm.
    """
    def __init__(self, num_classes=25):
        super(DeepChordCRNN, self).__init__()
        
        # --- 1. FEATURE EXTRACTOR (ResNet Style with Spatial Dropout) ---
        self.cnn_init = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Block 1
        self.res_block1 = ResidualBlock(64, 64)
        self.pool1 = nn.MaxPool2d((1, 2))
        self.spatial_drop1 = nn.Dropout2d(p=0.1) # Drops entire feature maps to force robustness
        
        # Block 2
        self.res_block2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d((1, 2))
        self.spatial_drop2 = nn.Dropout2d(p=0.2)
        
        # --- 2. TEMPORAL BRIDGE ---
        # 128 channels * 3 frequency bins remaining = 384
        lstm_input_size = 128 * 3
        
        # Pre-LSTM stabilizer to prevent NaNs
        self.pre_lstm_norm = nn.LayerNorm(lstm_input_size)
        
        # --- 3. SEQUENCE MODELING (Heavy Bi-LSTM) ---
        self.rnn = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=256, 
            num_layers=3, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.4
        )
        
        # --- 4. ROBUST CLASSIFIER ---
        # 256 * 2 (bidirectional) = 512
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),       # Post-LSTM stabilizer
            nn.Linear(512, 128), 
            nn.GELU(),               # More fluid and stable than standard ReLU
            nn.Dropout(0.4), 
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 1. Spatial extraction
        x = self.cnn_init(x)
        
        x = self.res_block1(x)
        x = self.pool1(x)
        x = self.spatial_drop1(x)
        
        x = self.res_block2(x)
        x = self.pool2(x)
        x = self.spatial_drop2(x)
        
        # Prepare for LSTM: flatten frequency bins into channels
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, -1)
        
        # 2. Temporal modeling
        x = self.pre_lstm_norm(x)
        rnn_out, _ = self.rnn(x)
        
        # 3. Classification
        out = self.classifier(rnn_out)
        return out
